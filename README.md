### Introduction

Adaptive Resonance Theory, or ART, has been developed as a model to explain brain mechanisms such as rapid categorisation of objects, remembering information over very long time ranges, and balancing between expected and unexpected information. ART neural networks possess some unique properties such as fast learning, resistance to catastrophic forgetting, and an incremental architecture which allows the insertion of new neurons on demand.

TopoART combines these properties with the ability of topology-learning neural networks to associate related neurons in order to represent the topology of the input. This knowledge can be exploited in various ways, e.g., clusters of arbitrary shapes can be learnt, the spatio-temporal context of input can be used to facilitate network training, and episodes can be formed. Details can be found here:

Marko Tscherepanow (2010). [_TopoART: A topology learning hierarchical ART network_](https://pub.uni-bielefeld.de/publication/1925596). In Proceedings of the International Conference on Artificial Neural Networks (ICANN), LNCS 6354, pp. 157–167. Berlin, Germany: Springer.

Marko Tscherepanow, Marco Kortkamp, and Marc Kammer (2011). [_A Hierarchical ART Network for the Stable Incremental Learning of Topological Structures and Associations from Noisy Data_](https://pub.uni-bielefeld.de/publication/2279367). Neural Networks 24(8): 906-916. Elsevier.

The combined properties of ART and topology-learning neural networks render TopoART neural networks particularly well suited to frequent problems arising in cognitive robotics and advanced machine learning, such as online-learning, lifelong learning from data streams, as well as incremental learning and prediction from non-stationary data, noisy data, imbalanced data, and incomplete data.

### LibTopoART

[LibTopoART](https://www.libtopoart.eu) provides implementations of several neural networks based on the TopoART architecture. This architecture has been developed as a unified machine learning approach tackling frequent problems arising in cognitive robotics and advanced machine learning, such as online-learning, lifelong learning from data streams, as well as incremental learning and prediction from non-stationary data, noisy data, imbalanced data, and incomplete data.

Besides the TopoART neural network itself, LibTopoART comprises implementations of Episodic TopoART (episodic clustering of data streams), Hypersphere TopoART (clustering and topology-learning), TopoART-AM (associative memory), TopoART-C and Hypersphere TopoART-C (classification), and TopoART-R (regression).