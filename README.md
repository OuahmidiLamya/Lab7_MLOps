# Lab 7 : Gestion du cycle de vie des modèles avec MLflow

## Étape 1 : Initialisation de l’environnement et installation de MLflow

Cette étape consiste à mettre en place un environnement Python isolé afin de garantir la reproductibilité du projet MLOps et d’éviter les conflits de dépendances. Un environnement virtuel est créé puis activé, avant l’installation de MLflow, outil central pour la gestion du cycle de vie des modèles. Cette configuration permet ensuite de suivre les expérimentations, d’enregistrer les métriques, les paramètres et les artefacts des modèles de manière structurée et traçable.

<img width="1919" height="790" alt="image" src="https://github.com/user-attachments/assets/da045b01-1f16-4fea-900c-30ca7211474a" />

## Étape 2 : Création explicite de l’espace de stockage MLflow

La création explicite de l’espace de stockage MLflow consiste à structurer dès le départ un répertoire dédié aux artefacts du projet. Le dossier `mlflow/artifacts` est créé à la racine afin d’isoler clairement les fichiers produits lors des expérimentations (modèles entraînés, métriques, sorties de validation) des autres composants du code. Cette organisation facilite la traçabilité, la reproductibilité des expériences et la gestion du cycle de vie des modèles, tout en préparant une utilisation correcte du serveur de tracking MLflow.

<img width="1919" height="542" alt="image" src="https://github.com/user-attachments/assets/75a059f9-acbe-46cf-9076-825551d88535" />

## Étape 3 : Configuration du client MLflow

Cette étape consiste à configurer le client **MLflow** afin que tous les scripts du projet envoient automatiquement leurs expérimentations vers un **serveur de tracking centralisé**. L’URI du tracking server est définie via la variable d’environnement `MLFLOW_TRACKING_URI`, pointant vers l’adresse locale du serveur MLflow. Une vérification est ensuite effectuée pour s’assurer que la variable est correctement prise en compte. Cette configuration garantit une centralisation des métriques, paramètres et artefacts, facilitant le suivi et la comparaison des runs.

<img width="1919" height="265" alt="image" src="https://github.com/user-attachments/assets/ea99eb31-687c-4b98-9c6f-9fc3949ec2e4" />

## Étape 4 : Démarrage du serveur MLflow (tracking server)

Dans cette étape, un **serveur de tracking MLflow local** a été mis en place afin de centraliser le suivi des expérimentations et la gestion des modèles. Le serveur est configuré avec une base de données **SQLite** pour stocker les métadonnées (expériences, paramètres, métriques, versions de modèles) et un **artifact store dédié** pour conserver les fichiers générés (modèles entraînés, logs, artefacts). Lors du démarrage, MLflow initialise automatiquement la base de données `mlflow.db` et crée les tables nécessaires au suivi des runs. Une fois le serveur lancé, l’interface web MLflow devient accessible via l’adresse `http://127.0.0.1:5000`, permettant de visualiser les expériences, comparer les performances des modèles et gérer le registre des modèles. Cette configuration transforme le tracking server en **source centrale de vérité**, garantissant la traçabilité, la reproductibilité et la gouvernance du cycle de vie des modèles de Machine Learning.

<img width="1919" height="717" alt="image" src="https://github.com/user-attachments/assets/ac9ca4b3-faa2-44be-b543-746c7b46756c" />
<img width="585" height="168" alt="image" src="https://github.com/user-attachments/assets/1e94b394-2af3-458c-ad36-1e08b65917b6" />
<img width="1915" height="902" alt="image" src="https://github.com/user-attachments/assets/669d9bcf-f00b-41be-adfd-dbc5477fb858" />


## Étape 5 : Instrumentation réelle de train.py

Cette étape consiste à instrumenter le script `train.py` avec MLflow afin de tracer automatiquement chaque exécution d’entraînement. Lors du lancement du script, les paramètres (version, seed, seuil gate_f1), les métriques de performance (accuracy, precision, recall, F1), ainsi que le fichier du modèle exporté sont enregistrés dans le tracking server MLflow. Le pipeline entraîné est également publié dans le **Model Registry** sous le nom stable `churn_model`, ce qui permet de créer une nouvelle version du modèle à chaque exécution. Cette instrumentation transforme le script d’entraînement en un processus traçable, reproductible et comparable via l’interface MLflow.

<img width="1919" height="911" alt="image" src="https://github.com/user-attachments/assets/160b4c2b-fa1f-4519-b88f-6dc620adb984" />
<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/303c8bf7-8474-44ab-b2c3-12d30d7c7c3d" />


## Étape 6 : Observation du registry MLflow

Cette étape consiste à vérifier l’enregistrement effectif du modèle entraîné dans le **Model Registry MLflow**, qui centralise la gestion des versions de modèles. Après l’exécution de `train.py`, l’interface MLflow est consultée via l’URL `http://127.0.0.1:5000`, en accédant à l’onglet **Models**. Le modèle nommé **`churn_model`** y apparaît avec une **première version (Version 1)** automatiquement créée. Chaque version est associée à un run précis, permettant de retracer l’origine du modèle (paramètres, métriques, artefacts). Cette étape valide que le pipeline d’entraînement est correctement instrumenté et que MLflow joue désormais son rôle de **registre centralisé**, garantissant la traçabilité, la reproductibilité et la gestion du cycle de vie des modèles conformément aux principes du MLOps.

<img width="1919" height="451" alt="image" src="https://github.com/user-attachments/assets/9cbc802e-3efc-485f-84ab-7a835f50fa93" />
<img width="1919" height="618" alt="image" src="https://github.com/user-attachments/assets/68808adc-ce67-4d63-926a-354e7fb35d0c" />

## Étape 7 : Promotion d’un modèle (activation)

Cette étape introduit la notion de **promotion explicite d’un modèle** vers un environnement cible à l’aide du **Model Registry MLflow**. Un script dédié, `promote.py`, est utilisé afin de dissocier clairement la phase d’entraînement de la décision d’activation du modèle. Le script interroge le registry pour identifier la version la plus récente du modèle **`churn_model`**, puis lui assigne l’alias **`production`**. Cette action marque officiellement la version sélectionnée comme modèle actif, sans modifier ni relancer l’entraînement. L’alias apparaît immédiatement dans l’interface MLflow, garantissant une traçabilité complète entre les versions, les runs et l’état opérationnel du modèle. Cette étape illustre un principe fondamental du MLOps : **le déploiement devient une décision contrôlée, réversible et indépendante de l’apprentissage**, facilitant ainsi la gouvernance et la maintenance des modèles en production.

<img width="1918" height="204" alt="image" src="https://github.com/user-attachments/assets/1d430558-2579-4af0-8ee8-dfa7fec6a8d3" />
<img width="1919" height="349" alt="image" src="https://github.com/user-attachments/assets/4aa1d32d-56b1-40fe-8af4-318548a0ed4d" />

## Étape 8 : Rollback via MLflow Model Registry

Cette étape met en œuvre un mécanisme de **retour arrière (rollback) contrôlé** en s’appuyant exclusivement sur le **Model Registry MLflow**, qui devient désormais l’unique source de vérité pour la gestion des versions du modèle. Le script `rollback.py` ne manipule plus de fichiers locaux tels que `current_model.txt`, mais agit directement sur l’alias **`production`** du modèle **`churn_model`**. Il permet soit d’activer explicitement une version donnée, soit d’effectuer automatiquement un rollback vers la version immédiatement précédente de celle actuellement en production. Cette approche garantit une gestion robuste et traçable des modèles, totalement indépendante du code applicatif et des artefacts locaux. Grâce à l’utilisation des alias MLflow, le changement de modèle actif devient instantané, réversible et sans redéploiement, illustrant un principe clé du MLOps moderne : **la maîtrise du cycle de vie des modèles par des opérations explicites, sûres et auditables**.

<img width="1919" height="629" alt="image" src="https://github.com/user-attachments/assets/0dd18381-42c9-44fc-869e-8fb7e532bfe2" />
<img width="1918" height="385" alt="image" src="https://github.com/user-attachments/assets/8ba64a23-4596-49cd-b5f1-3f18e5e0621d" />
<img width="1919" height="390" alt="image" src="https://github.com/user-attachments/assets/64c676c9-166d-420c-b1ff-fff80a00793f" />

## Étape 9 : API : chargement du modèle actif

Cette étape consiste à **connecter dynamiquement l’API au Model Registry MLflow** afin qu’elle serve en permanence le **modèle actuellement actif**, identifié par l’alias **`production`**. L’API ne dépend plus d’un fichier local ni d’un chemin figé vers un modèle, mais interroge directement MLflow pour déterminer la version en cours. Lors du démarrage ou lors d’un appel, le modèle est chargé à partir de l’URI logique `models:/churn_model@production`, garantissant ainsi que toute promotion ou rollback effectué dans le registry est immédiatement pris en compte par l’API, sans redéploiement. Ce mécanisme assure une séparation claire entre la logique applicative et la gestion du cycle de vie des modèles, tout en offrant une forte robustesse opérationnelle. L’API devient alors un simple consommateur du modèle actif, ce qui constitue un principe fondamental d’une architecture MLOps industrielle.
<img width="1919" height="174" alt="image" src="https://github.com/user-attachments/assets/2f80976f-6b62-4c47-ad55-8fb16fbb30cb" />
<img width="1919" height="233" alt="image" src="https://github.com/user-attachments/assets/78057d9e-86ac-4463-a7ce-9622b07a71ba" />

