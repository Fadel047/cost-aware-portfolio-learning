# Cost-Aware Portfolio Learning

## Contexte

La gestion d’un portefeuille implique de choisir une allocation entre plusieurs actifs. Les décisions d’investissement entraînent toutefois des coûts de transaction, qui peuvent réduire les gains attendus, en particulier lorsque le portefeuille est rééquilibré fréquemment.

Les méthodes classiques apprennent d’abord à prévoir les rendements, puis utilisent ces prévisions pour construire un portefeuille. Cette approche ne tient pas toujours compte directement de l’effet des décisions et des coûts associés.

## Problématique

**L’apprentissage directement orienté vers la décision permet-il de construire des portefeuilles plus pertinents que l’approche classique prédiction puis optimisation, lorsque les coûts de transaction sont pris en compte ?**

## Travail réalisé

Ce projet étudie et compare plusieurs approches de construction de portefeuille :

- des stratégies de référence simples, comme l’équipondération et la minimisation de la variance ;
- une approche classique qui prédit les rendements avant d’optimiser l’allocation ;
- un prototype d’apprentissage orienté vers la décision (*decision-focused learning*), qui ajuste le modèle en fonction d’une fonction de perte liée aux décisions de portefeuille.

Les méthodes sont évaluées dans un cadre commun, avec des contraintes d’allocation et une prise en compte des coûts de transaction. Le projet comprend également des tests portant notamment sur le calendrier des données, les contraintes d’optimisation et le calcul des gradients du prototype.

## Conclusion

La problématique a été étudiée, mais elle n’est **pas résolue positivement par le prototype actuel** : le travail ne permet pas de conclure que l’apprentissage orienté vers la décision améliore la construction de portefeuille par rapport à l’approche classique, une fois les coûts de transaction pris en compte.

Cette conclusion concerne uniquement les méthodes et le cadre étudiés dans ce projet. Une évaluation complémentaire, fondée sur un protocole de validation défini à l’avance et des données de test non réutilisées, serait nécessaire pour établir si cette approche peut apporter un bénéfice.

## Installation et exécution

Python 3.11 ou une version ultérieure est requis. Depuis la racine du projet :

```bash
python -m pip install -r requirements.txt
python src/download_data.py
python src/analyze_data.py
python src/diagnose_pto.py
python src/decision_focused.py
python -m unittest discover -s tests
```

Les scripts `backtest_baselines.py`, `predict_then_optimize.py` et `tune_pto.py` permettent d’exécuter des expériences complémentaires. `tune_pto.py` effectue une recherche de paramètres et modifie `config/pto_selected.json`.

Les données et les résultats générés ne sont pas suivis par Git. Si vous disposez déjà de l’instantané de données fourni avec le projet, vous pouvez ignorer les étapes de téléchargement et d’analyse.

## Organisation du projet

- `src/` : scripts d’analyse, de prévision, d’optimisation et d’apprentissage orienté vers la décision ;
- `config/` : configurations des expériences ;
- `tests/` : tests automatisés.

Pour lancer uniquement les tests :

```bash
python -m unittest discover -s tests
```
