import os
import time

from dataset import get_dataset, get_evaluation_labeled_data
from model import get_model
from utils import get_args_parser, print_config, read_config, seed_all
from vespa_utils.app import get_vespa_app
from vespa_utils.convert_feed import send_dataset_to_vespa_app
from vespa_utils.metric import compute_metrics, set_metrics
from vespa_utils.query import get_vespa_query_models
from vespa_utils.schema import schema


def main(args, config):
    # make config visible to user
    print_config(config)

    # set seed to making reproducible results
    seed_all(args.seed)

    # send data to vector database
    app = get_vespa_app(config["app_name"], schema)
    model = get_model(config["model_name"])
    dataset = get_dataset(config["dataset"])
    send_dataset_to_vespa_app(app, dataset, model, config["app_name"])

    # get test dataset and evaluation metric
    labeled_data = get_evaluation_labeled_data(dataset)
    eval_metrics = set_metrics(config["metrics"])
    start_time = time.time()

    # configure vespa QueryModel including ranking info
    query_models = get_vespa_query_models(config["model_name"], model)

    # perform the evaluation
    result = compute_metrics(app, labeled_data, eval_metrics, query_models)

    # measure query time: time between send a query and returned results
    end_time = time.time()
    elapsed_time = end_time - start_time
    avg_query_time = round(elapsed_time / len(labeled_data), 4)
    print(f"Average query time: {avg_query_time} seconds")

    # print the results in the terminal
    print(result.to_string())

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = f'{config["app_name"]}_{config["model_name"]}_{config["dataset"]}.csv'
    file_path = args.output_dir + output_name
    result.to_csv(file_path)


if __name__ == "__main__":
    # get config
    parser = get_args_parser()
    args = parser.parse_args()

    if args.config_file:
        config = read_config(args.config_file)
    else:
        config = read_config("./configs/txt2img.yaml")

    main(args, config)
