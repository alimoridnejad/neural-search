from vespa.package import HNSW, Document, Field, FieldSet, RankProfile, Schema

document = Document(
    fields=[
        Field(
            name="title",
            type="string",
            indexing=["index", "summary"],
            index="enable-bm25",
        ),
        Field(
            name="body",
            type="array<string>",
            indexing=["index", "summary"],
            index="enable-bm25",
        ),
        Field(
            name="id", type="string", indexing=["summary", "attribute"], rank="filter"
        ),
        Field(
            name="text_embedding",
            type="tensor<float>(p{},x[512])",
            indexing=["attribute", "index"],
            ann=HNSW(
                distance_metric="euclidean",
                max_links_per_node=16,
                neighbors_to_explore_at_insert=200,
            ),
        ),
        Field(
            name="image_embedding",
            type="tensor<float>(i{},x[512])",
            indexing=["attribute", "index"],
            ann=HNSW(
                distance_metric="euclidean",
                max_links_per_node=16,
                neighbors_to_explore_at_insert=200,
            ),
        ),
    ]
)


schema = Schema(
    name="document_schema",
    document=document,
    fieldsets=[
        FieldSet(
            name="default",
            fields=["title", "body"],
        )
    ],
    rank_profiles=[
        RankProfile(  # text query on normal text
            name="rank_bm25", inherits="default", first_phase="bm25(title) + bm25(body)"
        ),
        RankProfile(
            name="rank_text_data_embed",  # text or image query on text docs embed
            inherits="default",
            first_phase="closeness(text_embedding)",
            inputs=[("query(query_embed)", "tensor<float>(x[512])")],
        ),
        RankProfile(
            name="rank_image_data_embed",  # text or image query on image data embed
            inherits="default",
            first_phase="closeness(image_embedding)",
            inputs=[("query(query_embed)", "tensor<float>(x[512])")],
        ),
        RankProfile(
            name="rank_hybrid",  # text query on text data embed and normal text
            inherits="default",
            first_phase="bm25(title) + bm25(body) + closeness(text_embedding)",
            inputs=[("query(query_embed)", "tensor<float>(x[512])")],
        ),
    ],
)
