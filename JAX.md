# ¿Qué es JAX?

* JAX es una biblioteca de Python desarrollada por Google para la computación numérica y el aprendizaje automático es decir para todo aquello que tenga que ver con la IA. Su gran distinción se basa en la capacidad que tiene de hacer cálculos de forma muy eficiente en todo tipo de procesadores (CPU, GPU y TPU) mediante calculos/transformaciones funcionales avanzadas. Nació como una evolución de Autograd, con una integración más profunda en operaciones matriciales y optimización computacional.

## Principales características

* Diferenciación Automática (Autograd mejorado)

    * Permite calcular gradientes de funciones de manera eficiente, incluso para operaciones complejas y estructuras de datos arbitrarias.

    * Soporta diferenciación de orden superior, lo que lo hace ideal para redes neuronales y optimización matemática avanzada.

* Compilación Just-In-Time (JIT) con XLA
    * Usa XLA (Accelerated Linear Algebra) para compilar código en una versión optimizada antes de ejecutarlo, mejorando el rendimiento drásticamente.

    * Convierte funciones Python en operaciones altamente optimizadas en hardware acelerado.

* Ejecuta en CPU, GPU y TPU sin cambios en el código

    * Cambiar entre CPU, GPU y TPU es sencillo gracias a la API jax.device_put() y jax.jit() sin necesidad de escribir código específico para cada hardware.

* Transformaciones Funcionales Avanzadas

    * **jit(f)**: Compila y acelera funciones automáticamente.
    * **grad(f)**: Calcula derivadas y gradientes de forma automática.
    * **vmap(f)**: Vectoriza funciones para procesar múltiples datos en paralelo.
    * **pmap(f)**: Paraleliza cálculos en múltiples dispositivos.

* Compatibilidad con NumPy

    * Ofrece una API similar a NumPy (jax.numpy) con la ventaja de ejecutar operaciones en hardware acelerado sin modificar el código de NumPy.

* Ecosistema emergente

    * Se integra con librerías como Flax (para redes neuronales) y Optax (para optimización), convirtiéndolo en una alternativa viable a TensorFlow y PyTorch en investigación y producción.

## Comparación de JAX con TensorFlow y PyTorch.

### 1. Filosofía y Diseño  
| Característica | JAX | TensorFlow | PyTorch |
|--------------|----|------------|--------|
| **Enfoque** | Computación funcional con transformaciones automáticas (JIT, autodiff, vectorización). | Computación declarativa con gráficos estáticos/dinámicos y optimización con XLA. | Computación imperativa con gráficos dinámicos (define-by-run). |
| **Paradigma** | Basado en NumPy con diferenciación automática. | Define y optimiza gráficos computacionales (TensorFlow 2.0 usa Eager Execution). | Orientado a programación dinámica y código intuitivo para investigadores. |
| **Curva de Aprendizaje** | Más compleja por su paradigma funcional. | Compleja en versiones antiguas, más sencilla en TF 2.0. | Más intuitivo y fácil de usar. |

### 2. Ecosistema y Herramientas  
| Característica | JAX | TensorFlow | PyTorch |
|--------------|----|------------|--------|
| **Ecosistema** | Flax (redes neuronales), Optax (optimización), Haiku (DeepMind), TFP (probabilidad). | Amplia integración con Keras, TF-Probability, TF-Serving. | PyTorch Lightning, Hugging Face Transformers, TorchServe. |
| **Producción** | Menos maduro en producción, pero gana adopción en investigación. | Muy usado en producción (Google, empresas). | Cada vez más usado en producción, fuerte en investigación. |
| **Facilidad de integración** | Compatible con NumPy, pero requiere entender JIT y PMAP. | Compatible con muchas herramientas de ML y Cloud. | Fácil de usar con herramientas de investigación y frameworks populares. |

### 3. Diferenciación Automática y Computación en Hardware  
| Característica | JAX | TensorFlow | PyTorch |
|--------------|----|------------|--------|
| **Diferenciación automática** | Basado en Autograd con soporte para diferenciación de orden superior. | Autodiff con soporte para gráficos estáticos y dinámicos. | Autograd define gradientes automáticamente con gráficos dinámicos. |
| **Compilación JIT** | Usa XLA para optimizar código antes de la ejecución. | Usa XLA, pero más integrado en TF 2.0. | No tiene JIT integrado, pero usa TorchScript para optimización. |
| **Soporte en GPU/TPU** | Nativo y automático con `jax.jit()`. | Requiere configuración explícita, pero bien soportado. | Soporte nativo en GPU con CUDA y optimización con TorchScript. |

### Conclusión  
- **JAX** es la mejor opción si se busca una diferenciación automática avanzada y optimización computacional extrema.  
- **TensorFlow** es ideal para producción en entornos empresariales y escalabilidad en la nube.  
- **PyTorch** es más intuitivo y ampliamente adoptado en investigación, especialmente en NLP y visión por computadora.

## Ecosistema: librerías implementadas sobre JAX y otras herramientas que se integran bien con esta tecnología.

### 1. Librerías para Redes Neuronales  
| Librería | Descripción |
|----------|------------|
| **Flax** | Framework flexible y modular para construir modelos de deep learning con JAX. Similar a PyTorch en su facilidad de uso. |
| **Haiku** | Biblioteca desarrollada por DeepMind que sigue el paradigma de TensorFlow Sonnet. Diseñada para la investigación en IA. |
| **Elegy** | Framework inspirado en Keras, proporciona una API de alto nivel para definir y entrenar modelos en JAX. |

---

### 2. Optimización y Gradientes  
| Librería | Descripción |
|----------|------------|
| **Optax** | Biblioteca optimizada para la optimización de modelos de deep learning. Equivalente a `torch.optim` en PyTorch. |
| **JAXopt** | Conjunto de algoritmos de optimización como descenso de gradiente y programación cuadrática. Ideal para problemas científicos y de optimización matemática. |

---

### 3. Probabilidad y Estadística  
| Librería | Descripción |
|----------|------------|
| **TensorFlow Probability (TFP-JAX)** | Implementación de modelos probabilísticos y estadísticos en JAX. |
| **NumPyro** | Probabilistic programming basado en Pyro, diseñado para modelado bayesiano escalable. |

---

### 4. Herramientas Complementarias  
| Herramienta | Descripción |
|------------|------------|
| **Chex** | Conjunto de utilidades para pruebas y depuración en JAX. |
| **Jraph** | Biblioteca de DeepMind para aprendizaje en grafos con JAX. |
| **Diffrax** | Herramienta para resolver ecuaciones diferenciales con JAX. |

---

### 5. Integración con Otras Tecnologías  
- **NumPy**: JAX proporciona `jax.numpy`, una API similar a NumPy pero con soporte para hardware acelerado.
- **TPU/GPU**: JAX facilita la ejecución en hardware acelerado sin cambios en el código.
- **ML Frameworks**: Compatible con frameworks de IA como Hugging Face para el entrenamiento de modelos.


## WebGrafía

https://eiposgrados.com/blog-python/jax-machine-learning/

https://phuijse.github.io/tutorial_jax/README.html

https://mlearninglab.com/2019/04/14/autograd-una-de-las-claves-de-la-flexibilidad-de-pytorch/

https://www.reddit.com/r/programacion/comments/1i17igi/gu%C3%ADa_de_los_fundamentos_de_jax_para/?rdt=45159

https://cloud.google.com/blog/products/ai-machine-learning/guide-to-jax-for-pytorch-developers?utm_source=resumen_tech&utm_medium=email&utm_campaign=newsletter

https://www.computerworld.es/article/2115282/tensorflow-pytorch-y-jax-los-principales-marcos-de-deep-learning.html

https://iartificial.blog/aprendizaje/jax-vs-tensorflow-frameworks-de-computacion-numerica-comparados/

https://myscale.com/blog/es/jax-vs-pytorch-comprehensive-comparison-deep-learning/