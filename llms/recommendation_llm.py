
# pip install llama_index llama_index.embeddings.huggingface llama_index.llms.huggingface chromadb llama-index-vector-stores-chroma

import json
import chromadb
import time

from offerings import *
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import pipeline
from pprint import pprint

# from huggingface_hub import notebook_login

# import huggingface_hub
# huggingface_hub.login(token="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")


class RecommendationModel:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(RecommendationModel, cls).__new__(cls)
        return cls.instance
    
    def __init__ (self):
        self.TIME = time.time()

        self.EMBED_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
        self.EMBED_MODEL = HuggingFaceEmbedding(
            model_name=self.EMBED_MODEL_NAME,
            text_instruction="Given are the offers we provide, where each offer is uniqely identified by its offering_id",
            query_instruction="Retrieve all the relevent offering_ids from the given query",
            cache_folder="./models/embedding_models/"+str(hash(self.EMBED_MODEL_NAME))
        )


        self.CLASSIFIER_NAME = "facebook/bart-large-mnli"
        self.CLASSIFIER = pipeline(
            task="zero-shot-classification",
            model=self.CLASSIFIER_NAME,
            # model="facebook/bart-large-mnli"
            # model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0"   # is very large model
            # model="cross-encoder/nli-roberta-base"
            # model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"  # clothing for men
            # model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0"    # current best  
        )
        self.CLASSIFIER.save_pretrained(save_directory="./models/llm_models/"+str(hash(self.CLASSIFIER_NAME)))


        # self.PERSIST_DIR = "./vector-indexes/" + str(self.EMBED_MODEL.model_name)
        self.CHROMA_DB = chromadb.PersistentClient()
        self.CHROMA_COLLECTION = self.CHROMA_DB.get_or_create_collection(
            name="Categories",
            metadata={"hnsw:space": "cosine"}
        )

        self.notifyMessage("Model Initialized in "+str(time.time()-self.TIME)+"s")


    # Custom printer
    def notifyMessage(self, text):
        print(f"\n{'='*20}{text}{'='*20} ")

    # Classifier
    def generateResponseFromClassifier(self, userPrompt:str, threshold=0.51):
        self.TIME = time.time()
        return_ans = {
            "tags": [],
            "subtags": []
        }

        offer_subtags_array = []

        # Classify the userPrompt into Tags, and add their subtags.
        tag_ans = self.CLASSIFIER(userPrompt, OFFER_TAGS_ARRAY, multi_label=True)
        for label, score in zip(tag_ans['labels'], tag_ans['scores']):
            if score >= threshold:
                # Adding the tags to the answer
                return_ans['tags'].append(label)

                # Adding the subtags associated with the current tag into a temp array
                offer_subtags_array += list(OFFER_TAG_SUBTAG_DICT[label])


        # If the identified tags length is 0, then take all the subtags
        if len(return_ans['tags']) == 0:
            for tag in OFFER_TAGS_ARRAY:
                offer_subtags_array += list(OFFER_TAG_SUBTAG_DICT[tag])
                    
        # If tags are identified, classify the userPrompt into subtags
        if len(offer_subtags_array) > 0:
            subtag_ans = self.CLASSIFIER(userPrompt, offer_subtags_array, multi_label=True)
            for label, score in zip(subtag_ans['labels'], subtag_ans['scores']):
                if score >= threshold:
                    # Adding the subtags to the answer
                    return_ans['subtags'].append(label)



        self.notifyMessage("Classified in "+str(time.time()-self.TIME)+"s")
        print("Classified Labels: ", return_ans)
        return return_ans



    # Create stand alone ChromaDB vector store
    # Will recreate all the embeddings
    def createSimpleChromaDB(self):
        file = open("./categories.json", "r")
        self.nodes = json.load(file)
        self.notifyMessage("File Read")
    
        counter=1

        for i in self.nodes["data"]:
            counter+=1

            curr_metadata = {"offering_id": "", "tag": "", "subtags": ""}
            curr_id = i["offering_id"]
            curr_metadata["offering_id"] = i["offering_id"]
            json_str = json.dumps(i)

            curr_embeddings = self.EMBED_MODEL.get_text_embedding(json_str)

            for tag_i in i["tag"]:
                curr_metadata["tag"] = str(tag_i["name"])
                for sub_i in tag_i["subtags"]:
                    curr_metadata["subtags"] = str(sub_i["name"])

            self.CHROMA_COLLECTION.add(
                documents=[json_str],
                embeddings=[curr_embeddings],
                metadatas=[curr_metadata],
                ids=[curr_id]
            )

            if counter%50 == 0:
                print("parsed ", counter, " nodes")


    # Query the ChromaDB vector store
    def querySimpleChromaDB(self, userPrompt, top_res=5):
        # self.notifyMessage("Querying ChromaDB")
        self.TIME = time.time()

        embedded_text = self.EMBED_MODEL.get_query_embedding(userPrompt)

        classified_labels_dict = self.generateResponseFromClassifier(userPrompt)

        clause = {"$or": [
            
        ]}

        if classified_labels_dict["tags"] != []:
            clause["$or"].append({"tag": {"$in": classified_labels_dict["tags"]}})
        if classified_labels_dict["subtags"] != []:
            clause["$or"].append({"subtags": {"$in": classified_labels_dict["subtags"]}})
        if len(clause["$or"])!=2:
            clause = None
            
        print("clause:  ", clause)

        ans = self.CHROMA_COLLECTION.query(
            query_embeddings=[embedded_text],
            n_results=top_res,
            include=["metadatas", "documents"],
            where=clause
        )    

        self.notifyMessage(f"Query Time: {time.time()-self.TIME}s")

        print(ans["ids"],"\n", ans["metadatas"])

        docs_arr=[]
        for i in ans["documents"][0]:
            docs_arr.append(json.loads(i))

        return {"ids": ans["ids"][0], "documents": docs_arr}


if __name__ == "__main__":
    print("Starting Locally")
    
    l = RecommendationModel()

    while(True):
        input_prompt = input("What are you looking for (Empty for default prompt)? ")
        if input_prompt == "":
            input_prompt = "I am going to a party and I want to surprise my friend with a sweet dish I prepared for him."
        if(input_prompt == "exit"):
            break
        
        print("Prompt: ", input_prompt)
        id_array, docs_array = l.querySimpleChromaDB(input_prompt, 5)



