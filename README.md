# psoriasisGeneAnalysis

Rodar análise de proteínas:

cd biomarker_pipeline
python src/07_predict.py --model {modelo desejado}

- Analisa todos os arquivos FASTA em biomarker_pipeline/data/input

Modelos disponíveis:
- gradient_boosting
- knn
- logistic_regression > Apresentou melhor resultado nos testes
- random_forest
- naive_bayes