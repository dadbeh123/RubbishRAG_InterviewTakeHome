"""Run original RubbishRAG stages with parameters from tune_bench.py.

This module does NOT introduce a new ranking algorithm. It calls the project's
existing process_retrieval(), rerank_hits(), and resolve() functions.
"""
import json
import os


def install(config):
    from rubbish_rag import pipeline as P
    import rubbish_rag as R

    if config['fusion'] == 'lencomp':
        kw = {'gamma': config['gamma'], 'dense_weight': config['dense_weight']}
    elif config['fusion'] == 'rrf':
        kw = {'k': config['rrf_k'], 'weights': (1.0, config['rrf_dense_weight'])}
    else:
        kw = {}

    @R.retriever
    def tuned_retrieve(query, topk=5, api_key=None):
        # Server boundary is honored: all live candidates come from /retrieve.
        from server._hidden_retriever import remote_retrieve
        raw = remote_retrieve(P.normalize_fa(query),
                              topk=max(topk, config['pool_depth']), api_key=api_key)
        processed = P.process_retrieval(
            raw, topk=topk, strategy=config['fusion'],
            depth=config['rerank_depth'], ground=True, **kw
        )
        result = P.HitList(processed)
        result.topk = topk
        return result

    @R.reranker
    def tuned_rerank(query, hits):
        return P.rerank_hits(
            query, hits, mmr_lambda=config['mmr_lambda'],
            slot_beta=config['slot_beta'], dedup=config['dedup'],
            topk=getattr(hits, 'topk', None)
        )

    return tuned_retrieve, tuned_rerank


def install_from_env():
    path = os.getenv('RUBBISH_TUNE_CONFIG')
    if path:
        with open(path, encoding='utf-8') as f:
            install(json.load(f))
