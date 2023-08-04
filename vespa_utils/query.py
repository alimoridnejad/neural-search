from learntorank.query import QueryModel

from utils import logger


def create_vespa_query(query, model):
    """
    This method takes query as parameter and returns the body of a Vespa query
    """
    # number of results to be returned by vespa
    target_hits = 100

    # use proper rank profile and its field names from schema
    ranking_profile_name = "rank_image_data_embed"
    data_field_name = "image_embedding"
    query_field_name = "query_embed"

    # vespa uses yahoo query language (yql); build nearest neighbor part of it
    nn_part = f"nearestNeighbor({data_field_name},{query_field_name})"

    # get full yahoo query to be passed to vespa
    yql = f'select * from sources * where ({{"targetNumHits":{target_hits}}}{nn_part})'

    # get query embeddings using pre-trained models
    query_embed = model.encode_text(query)

    return {
        "yql": yql,
        "hits": target_hits,
        f"ranking.features.query({query_field_name})": query_embed,
        "ranking.profile": ranking_profile_name,
        "timeout": 10,
    }


def get_vespa_query_models(model_name, model):
    """
    This method returns a `QueryModel` which is an abstraction that encapsulates
    all the relevant information controlling how a Vespa app matches and ranks documents
    Returns:
    """
    logger.info("define vespa query model")
    query_models = [
        QueryModel(
            name=model_name, body_function=lambda x: create_vespa_query(x, model)
        )
    ]

    return query_models
