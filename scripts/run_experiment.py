import os
import json
import yaml
import random
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from textonomy.models.textonomy import Textonomy
from textonomy.utils.metrics import evaluate_textonomy_model
import matplotlib.pyplot as plt
import seaborn as sns
from textonomy.utils.plotting import plot_cluster_confusion_matrix
from dotenv import load_dotenv

def add_prefix_and_suffix_to_keys(d, prefix, suffix):
    return {f"{prefix}{key}{suffix}": value for key, value in d.items()}

def extract_keywords_from_df(df, text_col="tokenized_text", topic_col="supercategory", top_n=10):
    """
    Extracts top N keywords for each topic using TF-IDF.

    Args:
        df (pd.DataFrame): DataFrame with a text column and a topic label column.
        text_col (str): Name of the column containing document text.
        topic_col (str): Name of the column with topic labels.
        top_n (int): Number of keywords per topic.

    Returns:
        dict: {topic: [keyword1, keyword2, ...]} with top keywords for each topic.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    keywords_per_topic = {}

    for topic, group in df.groupby(topic_col):
        texts = group[text_col].astype(str).tolist()  # Ensure it's a list of strings
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10_000)
        tfidf_matrix = vectorizer.fit_transform(texts)

        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.mean(axis=0).A1  # Average TF-IDF per word

        top_keywords = [feature_names[i] for i in tfidf_scores.argsort()[-top_n:][::-1]]
        keywords_per_topic[topic] = top_keywords

    return keywords_per_topic


def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.
    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def run_experiment(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup
    load_dotenv(dotenv_path='.env')
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # MLflow setup
    mlflow_online = config.get('mlflow_online', False)
    if mlflow_online:
        username = os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('MLFLOW_TRACKING_PASSWORD')
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:8080')
    else:
        tracking_uri = 'http://localhost:8080'
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = get_or_create_experiment(config['experiment_name'])

    # Load data
    dataset_name = config['dataset_name']
    text_column = config['text_column']
    gold_label = config['gold_label']

    # Load the full training data first
    train_data_path = f"data/{dataset_name}/train.metadata.jsonl"
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    train_data = pd.read_json(train_data_path, lines=True)
    
    # Check how big the training data sample must be
    taxonomy_train_data_size = config["taxonomy_train_data_size"]
    clf_train_data_size = config["clf_train_data_size"]

    # Sample our training data using set seed
    train_data_sampling_seed = config["train_data_sampling_seed"]
    if clf_train_data_size > taxonomy_train_data_size:
        train_data = train_data.sample(n=clf_train_data_size, random_state=train_data_sampling_seed)
    else:
        train_data = train_data.sample(n=taxonomy_train_data_size, random_state=train_data_sampling_seed)

    test_data = pd.read_json(f"data/{dataset_name}/test.metadata.jsonl", lines=True)
    test_docs = test_data[text_column].tolist()

    # Data splitting
    random_state_train_test_split = config.get('random_state_train_test_split', random.randint(0, 1000))

    if 'taxonomy_train_data_size' in config:
        taxonomy_subset = train_data[:taxonomy_train_data_size]
        taxonomy_generation_data, _ = train_test_split(taxonomy_subset, train_size=0.9999999, shuffle=True, random_state=random_state_train_test_split)
    else:
        taxonomy_generation_data, _ = train_test_split(train_data, train_size=config['taxonomy_sample_ratio'], shuffle=True, random_state=random_state_train_test_split)
    
    taxonomy_docs = taxonomy_generation_data[text_column].tolist()

    if 'clf_train_data_size' in config:
        classifier_subset = train_data[:clf_train_data_size]
        classifier_training_data, classifier_test_data = train_test_split(classifier_subset, train_size=config['model_params']['classifier_train_test_split_ratio'], shuffle=True, random_state=random_state_train_test_split)
    else:
        classifier_training_data, classifier_test_data = train_test_split(train_data, train_size=config['model_params']['classifier_train_test_split_ratio'], shuffle=True, random_state=random_state_train_test_split)

    classifier_docs = classifier_training_data[text_column].tolist()
    classifier_test_docs = classifier_test_data[text_column].tolist()


    for i in range(config['loops']):
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.set_tag("training_info", f"{i+1}/{config['loops']}: {config['train_tag_text']}")
            mlflow.openai.autolog()

            # Log dataset info
            dataset: mlflow.data.pandas_dataset.PandasDataset = mlflow.data.from_pandas(train_data, source="Local Storage", name=dataset_name)
            mlflow.log_input(dataset=dataset, context=dataset_name)
            mlflow.log_param('random_state_train_test_split', random_state_train_test_split)
            mlflow.log_param('total_train_sample_size', len(train_data))
            mlflow.log_param('tax_sample_size', len(taxonomy_docs))
            mlflow.log_param('clf_train_sample_size', len(classifier_docs))
            mlflow.log_param('clf_test_sample_size', len(classifier_test_docs))
            mlflow.log_param('test_sample_size', len(test_docs))

            # Model and params
            model = Textonomy(**config['model_params'])
            mlflow.log_params(model.get_params())
            mlflow.log_param('evaluation_model', config['evaluation_model'])
            
            # Model and params
            model = Textonomy(**config['model_params'])
            mlflow.log_params(model.get_params())
            mlflow.log_param('evaluation_model', config['evaluation_model'])

            ## 1. TAXONOMY GENERATION
            print("Starting the taxonomy generation pipeline...")
            taxonomy, taxonomies, taxonomy_generation_data['summary'], taxonomy_generation_data['explanation'] = model.taxonomy_generation_pipeline(raw_corpus_sample=taxonomy_docs)
            print("Taxonomy generation pipeline completed. Logging results to mlflow...")
            mlflow.log_metric('num_clusters', len(taxonomy.labels))
            mlflow.log_metric('batch_size', model.batch_size)

            # Log taxonomy to mlflow
            taxonomy_path = "results/tmp/taxonomy.json"
            os.makedirs(os.path.dirname(taxonomy_path), exist_ok=True)
            with open(taxonomy_path, "w") as f: 
                json.dump(taxonomy.model_dump(), f, indent=4)
            mlflow.log_artifact(local_path=taxonomy_path, artifact_path="taxonomy.json")
            os.remove(taxonomy_path)

            # Log taxonomies to mlflow
            taxonomies_path = "results/tmp/taxonomies.json"
            with open(taxonomies_path, "w") as f: 
                for taxonomy in taxonomies: 
                    json.dump(taxonomy.model_dump(), f, indent=4)
            mlflow.log_artifact(local_path=taxonomies_path, artifact_path="taxonomies.json")
            os.remove(taxonomies_path)
            
            # Log summaries of taxonomy generation data to mlflow
            taxonomy_generation_data_path = "results/tmp/taxonomy_generation_data.json"
            taxonomy_generation_data[['id', gold_label, 'summary', 'explanation']].to_json(taxonomy_generation_data_path, orient="records")
            mlflow.log_artifact(local_path=taxonomy_generation_data_path, artifact_path="taxonomy_generation_data_summaries.json")
            os.remove(taxonomy_generation_data_path)

            ## 2. CLASSIFIER TRAINING
            print("Starting the classifier training pipeline...")
            classifier, classifier_training_data['pseudo_label'], classifier_training_data['pseudo_label_explanation'], classifier_test_data['pseudo_label'], classifier_test_data['pseudo_label_explanation'] = model.classifier_training_pipeline(train_data=classifier_docs, test_data=classifier_test_docs)
            print("Classifier training pipeline completed. Logging results to mlflow...")

            # Log if there is an outlier cluster in the pseudo labelled data
            if classifier_training_data['pseudo_label'].nunique() > len(model.taxonomy.labels) or classifier_test_data['pseudo_label'].nunique() > len(model.taxonomy.labels):
                mlflow.log_param('outlier_cluster', True)
            else:
                mlflow.log_param('outlier_cluster', False)

            # Log classifier to mlflow
            mlflow.sklearn.log_model(model.classifier, "lr_classifier", signature=model.classifier_signature, input_example=model.classifier_input_example)
            mlflow.log_param('encoder_model_name', model.encoder_model_name)

            # Log classifier metrics
            mlflow.log_metric("clf_accuracy", float(model.accuracy_))
            mlflow.log_metric("clf_f1_score", float(model.f1_score_))
            # Save and log classification report
            report_df = pd.DataFrame(model.metrics_).transpose()
            report_path = "results/tmp/classification_report.csv"
            report_df.to_csv(report_path)
            mlflow.log_artifact(local_path=report_path, artifact_path="classification_report.csv")
            os.remove(report_path)
            
            # Confusion matrix for classifier
            conf_matrix_path_png = "results/tmp/confusion_matrix.png"
            plt.figure(figsize=(15,15))
            sns.heatmap(model.confusion_matrix_, annot=True, fmt="d", cmap="Blues", xticklabels=model.label_encoder.inverse_transform(model.classes_), yticklabels=model.label_encoder.inverse_transform(model.classes_))
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Classifier Confusion Matrix")
            plt.savefig(conf_matrix_path_png, bbox_inches='tight', dpi=300)
            mlflow.log_artifact(local_path=conf_matrix_path_png, artifact_path="confusion_matrix.png")
            os.remove(conf_matrix_path_png)
            conf_matrix_path_csv = "results/tmp/confusion_matrix.csv"
            pd.DataFrame(model.confusion_matrix_).to_csv(conf_matrix_path_csv)
            mlflow.log_artifact(local_path=conf_matrix_path_csv, artifact_path="confusion_matrix.csv")
            os.remove(conf_matrix_path_csv)

            ## 3. TEST
            # Test Textonomy on test data
            print("Starting the test pipeline...")
            test_data['predicted_label'] = model.predict(test_docs)
            print("Test pipeline completed. Logging results to mlflow...")
            internal_cluster_metrics = add_prefix_and_suffix_to_keys(model.internal_cluster_metrics, prefix="int_", suffix='_test')
            mlflow.log_metrics(internal_cluster_metrics)

            # Log clustered test data to mlflow
            clustered_test_data = "results/tmp/clustered_test_data.json"
            test_data[['id', gold_label, 'predicted_label']].to_json(clustered_test_data, orient="records")
            mlflow.log_artifact(local_path=clustered_test_data, artifact_path="clustered_test_data.json")
            os.remove(clustered_test_data)

            # Save cluster confusion matrix
            conf_matrix_path_png = "results/tmp/cluster_confusion_matrix.png"
            plot_cluster_confusion_matrix(model, test_data[gold_label], test_data['predicted_label'], conf_matrix_path_png)
            mlflow.log_artifact(local_path=conf_matrix_path_png, artifact_path="cluster_confusion_matrix.png")
            os.remove(conf_matrix_path_png)

            # Log Topic Coherence, Diversity and Usefulness metrics
            topic_words = extract_keywords_from_df(test_data, text_col='tokenized_text', topic_col='predicted_label', top_n=10)
            mlflow.log_dict(topic_words, "tf_idf_topic_words.json")
            topic_words_list = [i for i in topic_words.values()]
            ref_corpus_texts = [doc.split() for doc in test_data['tokenized_text'].to_list()]
            coh_div_metrics, llm_uf_reasoning = evaluate_textonomy_model(taxonomy=model.taxonomy, texts=ref_corpus_texts, use_case=model.use_case, model=config['evaluation_model'], topic_words=topic_words_list)
            mlflow.log_text(str(llm_uf_reasoning), "U_LLM_USEFULNESS_reasoning.txt")

            valid_metrics, invalid_metrics = {}, {}
            for k, v in coh_div_metrics.items():
                try:
                    valid_metrics[k] = float(v)
                except (ValueError, TypeError):
                    invalid_metrics[k] = v
            mlflow.log_metrics(valid_metrics)
            if invalid_metrics:
                invalid_metrics_path = "results/tmp/invalid_metrics.txt"
                with open(invalid_metrics_path, "w") as f:
                    json.dump(invalid_metrics, f, indent=4)
                mlflow.log_artifact(invalid_metrics_path)
                os.remove(invalid_metrics_path)

            # Test Texotonomy on taxonomy generation data
            print("Starting the test pipeline on taxonomy generation data...")
            taxonomy_generation_data['predicted_label'] = model.predict(taxonomy_docs)
            print("Test pipeline on taxonomy generation data completed. Logging results to mlflow...")
            internal_cluster_metrics = add_prefix_and_suffix_to_keys(model.internal_cluster_metrics, prefix="int_", suffix='_tax')
            mlflow.log_metrics(internal_cluster_metrics)

            # Log clustered taxonomy generation data to mlflow
            clustered_taxonomy_generation_data = "results/tmp/clustered_taxonomy_generation_data.json"
            taxonomy_generation_data[['id', gold_label, 'predicted_label']].to_json(clustered_taxonomy_generation_data, orient="records")
            mlflow.log_artifact(local_path=clustered_taxonomy_generation_data, artifact_path="clustered_taxonomy_generation_data.json")
            os.remove(clustered_taxonomy_generation_data)

            # Test classifier on classifier training data
            print("Starting the test pipeline on classifier training&test data...")
            classifier_training_data['predicted_label'] = model.predict(classifier_docs)
            internal_cluster_metrics = add_prefix_and_suffix_to_keys(model.internal_cluster_metrics, prefix="int_", suffix='_clf_train')
            mlflow.log_metrics(internal_cluster_metrics)
            classifier_test_data['predicted_label'] = model.predict(classifier_test_docs)
            internal_cluster_metrics = add_prefix_and_suffix_to_keys(model.internal_cluster_metrics, prefix="int_", suffix='_clf_test')
            mlflow.log_metrics(internal_cluster_metrics)
            print("Test pipeline on classifier training&test data completed. Logging results to mlflow...")
            
            # Log pseudo labelled data to mlflow
            pseudo_labelled_train_data = "results/tmp/pseudo_labelled_classifier_train_data.json"
            classifier_training_data[['id', gold_label, 'pseudo_label', 'predicted_label', 'pseudo_label_explanation']].to_json(pseudo_labelled_train_data, orient="records")
            mlflow.log_artifact(local_path=pseudo_labelled_train_data, artifact_path="pseudo_labelled_classifier_train_data.json")
            os.remove(pseudo_labelled_train_data)
            pseudo_labelled_test_data = "results/tmp/pseudo_labelled_classifier_test_data.json"
            classifier_test_data[['id', gold_label, 'pseudo_label', 'predicted_label', 'pseudo_label_explanation']].to_json(pseudo_labelled_test_data, orient="records")
            mlflow.log_artifact(local_path=pseudo_labelled_test_data, artifact_path="pseudo_labelled_classifier_test_data.json")
            os.remove(pseudo_labelled_test_data)

            ## SCORING - Calculate and log external cluster metrics on TEST data
            external_cluster_metrics = model.calculate_external_cluster_metrics(y_true=test_data[gold_label], y_pred=test_data['predicted_label'])
            external_cluster_metrics = add_prefix_and_suffix_to_keys(external_cluster_metrics, prefix="ext_", suffix='_test')
            mlflow.log_metrics(external_cluster_metrics)
            mlflow.log_param("gold_label", gold_label)

            # Calculate and Log external cluster metrics
            external_cluster_metrics = model.calculate_external_cluster_metrics(y_true=taxonomy_generation_data[gold_label], y_pred=taxonomy_generation_data['predicted_label'])
            external_cluster_metrics = add_prefix_and_suffix_to_keys(external_cluster_metrics, prefix="ext_", suffix='_tax')
            mlflow.log_metrics(external_cluster_metrics)

            # Ext Cluster metrics on train and test data
            # ext_X_clf_train_pseudo
            external_cluster_metrics = model.calculate_external_cluster_metrics(y_true=classifier_training_data[gold_label], y_pred=classifier_training_data['pseudo_label'])
            external_cluster_metrics = add_prefix_and_suffix_to_keys(external_cluster_metrics, prefix="ext_", suffix='_clf_train_pseudo')
            mlflow.log_metrics(external_cluster_metrics)
            # ext_X_clf_train_pred
            external_cluster_metrics = model.calculate_external_cluster_metrics(y_true=classifier_training_data[gold_label], y_pred=classifier_training_data['predicted_label'])
            external_cluster_metrics = add_prefix_and_suffix_to_keys(external_cluster_metrics, prefix="ext_", suffix='_clf_train_pred')
            mlflow.log_metrics(external_cluster_metrics)
            # ext_X_clf_test_pseudo
            external_cluster_metrics = model.calculate_external_cluster_metrics(y_true=classifier_test_data[gold_label], y_pred=classifier_test_data['pseudo_label'])
            external_cluster_metrics = add_prefix_and_suffix_to_keys(external_cluster_metrics, prefix="ext_", suffix='_clf_test_pseudo')
            mlflow.log_metrics(external_cluster_metrics)
            # ext_X_clf_test_pred
            external_cluster_metrics = model.calculate_external_cluster_metrics(y_true=classifier_test_data[gold_label], y_pred=classifier_test_data['predicted_label'])
            external_cluster_metrics = add_prefix_and_suffix_to_keys(external_cluster_metrics, prefix="ext_", suffix='_clf_test_pred')
            mlflow.log_metrics(external_cluster_metrics)

            print("Run completed. Ending run...")
            mlflow.end_run()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()
    run_experiment(args.config)
