# ImageToFlowmap

This project aims to generate flowmaps for usage in projection mapping or low cost river simulations-

The tool creates clusters of similar areas on the image selected, generates networks from distance-to-border-skeleton-generation and lets you edit the networks to generate a quasi-equipotential flowmpaps along geometries.

Uses multithreading but runs on the cpu and therefor is very slow. Start with a small image since complexity grows squared (512x512).
