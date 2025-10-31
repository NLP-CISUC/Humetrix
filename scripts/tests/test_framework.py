from humetrix import HumorAnalyzer


sentences = ['O que diz um coelho quando abre uma porta? Primeiro as cenouras.',
             'O que diz um coelho quando abre uma porta? Primeiro as senhoras.',
             'Qual é o youtuber que mais economiza na luz? O Jovem Led.',
             'Qual é o youtuber que mais economiza na luz? O Whindersson Nunes.',
             'Porque é que o computador não pára de espirrar? Porque apanhou um vírus.',
             'Porque é que o computador não pára de avariar? Porque apanhou um vírus.',
             'Qual o livro que conta a história dos imigrantes do sertão??? Vim das secas.',
             'Qual o livro que conta a história dos imigrantes do sertão??? O quinze',
             'O que é que acontece quando o Frodo morre? Passam-lhe uma certidão de Hobbit.',
             'O que é que acontece quando o Frodo morre? Passam-lhe uma certidão de óbito.']
signs = ['cenouras', 'cenouras',
         'led', 'led',
         'vírus', 'vírus',
         'vim das secas', 'vim das secas',
         'hobbit', 'hobbit']
alt_signs = ['senhoras', 'senhoras',
             'nerd', 'nerd',
             'vírus', 'vírus',
             'vidas secas', 'vidas secas',
             'óbito', 'óbito']
analyzer = HumorAnalyzer(language='pt',
                         embeddings_path='data/embeddings/pt/glove_s300.gensim',
                         skipgram_path='results/skipgram/pt.pt',
                         ngram_dir_path='data/ngrams/pt',
                         transformer_model_name='FacebookAI/xlm-roberta-base')

for sent, sign, alt_sign in zip(sentences, signs, alt_signs):
    qi_score = analyzer.quantum_incongruity(sent)
    qi_xlm_score = analyzer.quantum_incongruity(sent, backend='transformer')
    qu_score = analyzer.quantum_uncertainty(sent)
    qu_xlm_score = analyzer.quantum_uncertainty(sent, backend='transformer')
    lgs_score = analyzer.local_global_surprise(sent, sign, alt_sign)
    kao_amb_score = analyzer.kao_ambiguity(sent, sign, alt_sign)
    kao_dist_score = analyzer.kao_distinctiveness(sent, sign, alt_sign)
    print(f'Sentence: {sent}')
    print(f'Sign: {sign} | Alt sign: {alt_sign}')
    print(f'  QE-Incongruity GloVe: {qi_score:.4f}')
    print(f'  QE-Uncertainty GloVe: {qu_score:.4f}')
    print(f'  QE-Incongruity XLM-R: {qi_xlm_score:.4f}')
    print(f'  QE-Uncertainty XLM-R: {qu_xlm_score:.4f}')
    print(f'  Local-Global Surprise: {lgs_score:.4f}')
    print(f'  Kao et al. Ambiguity: {kao_amb_score:.4f}')
    print(f'  Kao et al. Distinctiveness: {kao_dist_score}')
    print('*****************')
