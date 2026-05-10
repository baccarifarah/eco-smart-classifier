# PROMPTS.md — Journal des interactions IA – TP7 NLP Eco-Smart

## Prompt 1 — Pipeline de prétraitement
**Objectif :** Nettoyer la colonne Rapport_Collecte.
**Prompt :** "Écris un pipeline NLP Python pour du texte technique français :
minuscules, suppression ponctuation, stopwords FR + domaine, stemming Snowball."
**Décision :** FrenchStemmer (Snowball) retenu car plus robuste que le lemmatiseur
NLTK FR sur les termes industriels (collecté → collect, contaminé → contamin).

---

## Prompt 2 — Pourquoi conserver les chiffres avant extraction
**Prompt :** "Pourquoi ne pas supprimer les chiffres (ex: 45.8 kg) avant Regex ?"
**Réponse intégrée :** Les expressions numériques portent une information
quantitative absente des colonnes structurées. On les supprime uniquement APRÈS
l'étape d'extraction Regex (Partie 2.2), pas avant.

---

## Prompt 3 — TF-IDF max_features
**Prompt :** "Quel max_features pour TF-IDF sur un dataset de ~500-1000 lignes ?"
**Décision :** max_features=500 limite la dimensionnalité et réduit le risque
de sur-apprentissage tout en conservant les bigrammes informatifs (ngram_range=(1,2)).

---

## Prompt 4 — Extraction Regex Contamination
**Prompt :** "Écris une regex Python pour détecter contamination/humidité/traces
dans des rapports techniques français."
**Décision :** Pattern multi-variantes avec \b pour éviter les faux positifs,
flags re.IGNORECASE pour la robustesse casse.

---

## Prompt 5 — Extraction état du matériau
**Prompt :** "Regex Python pour classer l'état d'un matériau en Neuf/Moyen/Brisé
depuis un texte libre français."
**Décision :** Priorité Brisé > Neuf > Moyen car les termes d'état dégradé sont
les plus discriminants pour la valeur de revente.

---

## Prompt 6 — Architecture fusion multimodale
**Prompt :** "Comment combiner une matrice sparse TF-IDF avec des colonnes numériques
dans scikit-learn sans perte d'efficacité mémoire ?"
**Décision :** scipy.sparse.hstack — conserve le format CSR,
compatible avec Ridge et LinearSVC sans conversion dense.

---

## Prompt 7 — MLflow tracking
**Prompt :** "Génère une fonction Python générique pour logger accuracy/F1/params
dans MLflow pour chaque couple (vectorisation, modèle)."
**Décision :** Un run par couple. Nommage : "Vecteur | Modèle" pour lisibilité UI.

---

## Tableau récapitulatif des choix techniques

| Choix | Valeur retenue | Justification |
|---|---|---|
| Stemmer | FrenchStemmer Snowball | Meilleur rappel sur termes techniques |
| TF-IDF max_features | 500 | Anti-surapprentissage |
| TF-IDF ngram_range | (1,2) | Capture bigrammes ("haute conductivité") |
| Word2Vec window | 5 | Adapté aux phrases techniques |
| Fusion | scipy.sparse.hstack | Efficacité mémoire (format CSR conservé) |
| Régression | Ridge alpha=1.0 | Robuste aux features corrélées |
