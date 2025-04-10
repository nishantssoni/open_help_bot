from openai import OpenAI
import os
import pandas as pd
from copy import deepcopy
from .utils import get_chat_response
from dotenv import load_dotenv
import json
load_dotenv()


class RecommendationAgent:
    def __init__(self,apriori_recommendation_path, popular_recommendation_path):
        self.client = OpenAI(
            api_key=os.getenv("TOKEN"),
            base_url=os.getenv("BASE_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

        with open(apriori_recommendation_path, "r") as f:
            self.apriori_recommendation = json.load(f)
        
        self.popular_recommendation = pd.read_csv(popular_recommendation_path)

        self.products = self.popular_recommendation["product"].tolist()
        self.products_categories = self.popular_recommendation["product_category"].tolist()
    
    def get_apriori_recommendation(self, products, top_k=5):
        recommendation_list = []
        

        for product in products:
            if product in self.apriori_recommendation:
                recommendation_list.extend(self.apriori_recommendation[product])
            
        # sort recommendation list by confidence
        recommendation_list = sorted(recommendation_list, key=lambda x: x['confidence'], reverse=True)

        recommendations = []
        recommendations_per_category = {}

        for recommendation in recommendation_list:
            if recommendation in recommendations:
                continue

            # limit 2 recommendation per category
            product_category = recommendation["product_category"]

            if product_category not in recommendations_per_category:
                recommendations_per_category[product_category] = 0

            if recommendations_per_category[product_category] >=2:
                continue

            recommendations_per_category[product_category] += 1

            # add recommendation
            recommendations.append(recommendation['product'])
            if len(recommendations) >= top_k:
                break

        return recommendations


    def get_popular_recommendation(self, product_category=None, top_k=5):
        popular_recommendation = self.popular_recommendation

        if type(product_category) == str:
            product_category = [product_category]

        if product_category is not None:
            popular_recommendation = self.popular_recommendation[self.popular_recommendation["product_category"].isin(product_category)]
        popular_recommendation = popular_recommendation.sort_values('count',ascending=False)

        if popular_recommendation.shape[0]==0:
            return []

        popular_recommendation = popular_recommendation['product'].head(top_k).tolist()
        return popular_recommendation