import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ragas import evaluate
from ragas.metrics import answer_relevancy, answer_similarity, answer_correctness, context_precision, context_recall, context_entity_recall
from datasets import Dataset

class RagaAnalizer:
    """
    Class to analyze responses generated based on questions using a RAG (Retrieval-Augmented Generation) model.
    """
    def __init__(self, retriever, rag_chain):
        """
        Initializes the parser with a document retrieval system and a response generation chain.
        """
        self.retriever = retriever
        self.rag_chain = rag_chain

    def preprocess_documents(self, splits):
        for document in splits:
            document.metadata['file_name'] = document.metadata['source']

    def infer_answers(self, questions):
        answers = [self.rag_chain.invoke(query) for query in questions]
        contexts = [[doc.page_content for doc in self.retriever.get_relevant_documents(query)] for query in questions]
        return questions, answers, contexts

    def create_dataset(self, questions, answers, contexts, ground_truths):
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truths": ground_truths
        }
        return Dataset.from_dict(data)

    def evaluate_answers(self, dataset):
        return evaluate(
            dataset=dataset,
            metrics=[
                answer_relevancy,
                answer_similarity,
                answer_correctness,
                context_precision,
                context_recall,
                context_entity_recall
            ],
        )

    def plot_results(self, results):
        df = results.to_pandas()
        heatmap_data = df[[
            "answer_relevancy",
            "answer_similarity",
            "answer_correctness",
            "context_precision",
            "context_recall",
            "context_entity_recall"
        ]]
        cmap = LinearSegmentedColormap.from_list('green_red', ['salmon', 'springgreen'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", linewidths=.5, cmap=cmap)
        plt.yticks(ticks=range(len(df['question'])), labels=df['question'], rotation=0)
        plt.show()
