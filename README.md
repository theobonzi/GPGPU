# 🖥️ Projet GPGPU - Séparation Fond / Objets Mobiles dans des Vidéos

## Contexte du Projet
Ce projet se concentre sur l'optimisation des opérations de traitement vidéo sur GPU. L'objectif est d'utiliser les capacités des GPU pour séparer efficacement les objets mobiles du fond dans des séquences vidéo.

## Données du Projet
Le projet traite des séquences vidéo standards, avec un modèle de fond initial qui est continuellement mis à jour pour identifier les objets mobiles. Les données vidéo sont traitées frame par frame.

## Objectifs du Projet
Les objectifs principaux du projet sont :
1. Réimplémenter une pipeline basé sur un article scientifique pour la séparation fond-objet.
2. Traiter un flux de trames vidéo en temps réel avec des techniques d'optimisation GPU.
3. Evaluer le projet sur la précision du code et la vitesse du framerate.

## Méthodologie
Le projet suit les étapes suivantes :
1. **Extraction des Caractéristiques** : Analyse des couleurs et textures des vidéos.
2. **Mesure de Similarité** : Application de méthodes de mesure de similarité pour distinguer le fond des objets mobiles.
3. **Classification** : Utilisation de l'intégrale de Choquet pour la classification des pixels.
4. **Optimisation GPU** : Mise en œuvre de CUDA pour le traitement sur GPU et amélioration des performances.

## Technologies Utilisées
- **Langages de programmation** : C++ (version CPU), CUDA (optimisation GPU).
