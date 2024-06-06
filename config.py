config = {
    'device': 'cuda',
    'exp_name': 'sao-semantics',
    'token_length': 512,
    'n_features': 2**14,
    'hidden_size': 768,
    'epochs': 1,
    'learning_rate': 1e-3,
    'lambda_reg': 5,
    'batch_size': 32,
    'accumulation_steps': 4,
    'warmup_steps': 200,
    'data': ('monology/pile-uncopyrighted',),
    'data': ('wikitext', 'wikitext-103-v1'),
    'data': ('allenai/c4', 'en'),
    'base_model': 'microsoft/deberta-v3-base',
}
