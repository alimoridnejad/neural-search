from learntorank import evaluation
from learntorank.evaluation import (MatchRatio,
                                    NormalizedDiscountedCumulativeGain, Recall,
                                    ReciprocalRank)

from utils import logger


def set_metrics(metric_names):
    """
    This method returns the list of metrics base on metric names.
    Returns: learntorank evaluation metrics.
    """

    vespa_metrics = []
    for name in metric_names:
        if name == "recall":
            vespa_metrics.append(Recall(at=1))
        elif name == "reciprocal_rank":
            vespa_metrics.append(ReciprocalRank(at=10))
        elif name == "ndcg":
            vespa_metrics.append(NormalizedDiscountedCumulativeGain(at=10))
        elif name == "match_ratio":
            vespa_metrics.append(MatchRatio())
        else:
            raise ValueError(f"Unsupported metric name: {name}")
    return vespa_metrics


def compute_metrics(app, labeled_data, eval_metrics, query_models):
    """
    This method returns evaluation results based on metrics, dataset, and query model.
    Args:
        app: vespa app
        labeled_data: pandas dataframe having specific fields
        eval_metrics: vespa learntorank evaluation metrics
        query_models: vespa query model

    Returns:
    """
    metric_names = []
    for metric in eval_metrics:
        metric_names.append(metric.name)
    logger.info(f"compute the metrics: {metric_names}. This might take a while ...")

    result = evaluation.evaluate(
        app=app,
        labeled_data=labeled_data,  # data containing query, query_id and relevant docs
        eval_metrics=eval_metrics,  # evaluation metrics
        query_model=query_models,  # query models to be evaluated
        id_field="id",  # the Vespa field representing the document id
    )

    return result
