# Tarea de LLM.
Mediante la integración de herramientas como LangChain, Pinecone y OpenAI, exploramos la intersección entre la inteligencia artificial y el procesamiento de lenguaje natural (NLP) en los desafíos propuestos. Para empezar, exploramos la interacción con ChatGPT utilizando Python y LangChain para enviar preguntas al modelo y obtener respuestas. Luego hablamos sobre la creación de un sistema de recuperación y generación de respuestas (RAG) utilizando una base de datos vectorial en memoria y Pinecone, destacando la eficiencia de la búsqueda de información. Finalmente, hablamos sobre cómo configurar otro RAG, esta vez utilizando Pinecone para administrar una base de datos vectorial en la nube, lo que demuestra la versatilidad de estas herramientas en la creación de sistemas de procesamiento de lenguaje natural avanzados.

## HERRAMIENTAS 
- [PYTHON](https://www.python.org/) : Lenguaje de programación manejado.
   <p align="center">
  <IMG src=https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png height=150 width=250 >
  <p/>
- [GIT](https://git-scm.com) : Para el manejo de las versiones.
  <p align="center">
  <IMG src=https://logowik.com/content/uploads/images/git6963.jpg height=150 width=250 >
  <p/>
  
## INSTALACIÓN 
+ Se clona el repositorio en una máquina local con el siguiente comando:
  ~~~
  git clone https://github.com/Juc28/TareaLLM-Arep.git
  ~~~
+ Entrar al directorio del proyecto con el siguiente comando:
    ```
    cd TareaLLM-Arep
    ```
+ Una vez clonado el proyecto se pude abrir desde un editor de código como Pycharm para editar porque se debe poner las siguientes: 

    * PINECONE_API_KEY --> Esta la obtenemos al crear una cuenta en https://www.pinecone.io/

      ![imagen](https://github.com/Juc28/TareaLLM-Arep/assets/118181224/b59d038a-1059-4c3f-8f92-358716856c93)


    * OPENAI_API_KEY --> Esta es proporcinada por el profesor 


  <img width="960" alt="pycharm64_Cv74NKyZfH" src="https://github.com/Juc28/TareaLLM-Arep/assets/118181224/e7a14368-9f52-4c8b-b1f0-487675e4fac1">

  
  ![imagen](https://github.com/Juc28/TareaLLM-Arep/assets/118181224/288e37c2-161a-4420-8486-2e8cd676eb8f)

+ Se dede instalar los paquetes que esta en el archivo [requirements.txt](https://github.com/Juc28/TareaLLM-Arep/blob/master/requirements.txt) con el siguiente comando:
  ~~~
  pip install -r requirements.txt
  ~~~
  
> *Para correr las clases desde consola o desde Pycharm:*
  
+ La primera clase [main.py](https://github.com/Juc28/TareaLLM-Arep/blob/master/main.py):
  ~~~
  py main.py
  ~~~
  ![imagen](https://github.com/Juc28/TareaLLM-Arep/assets/118181224/ea32ec66-0477-4579-86ab-93157ef87424)

  <img width="960" alt="J1" src="https://github.com/Juc28/TareaLLM-Arep/assets/118181224/c3c2ff80-138d-4046-bbcc-e152384774f6">
  
  
+ La segunda clase [memory.py](https://github.com/Juc28/TareaLLM-Arep/blob/master/memory.py):
   ~~~
  py memory.py
  ~~~

  <img width="960" alt="J2" src="https://github.com/Juc28/TareaLLM-Arep/assets/118181224/ed7d0cb9-bb18-4d12-aa23-d123f95d4750">
  
+ La tercera clase [pinecote.py](https://github.com/Juc28/TareaLLM-Arep/blob/master/pinecote.py):
  
  ~~~
  py pinecote.py
  ~~~

  <img width="960" alt="J2" src="https://github.com/Juc28/TareaLLM-Arep/assets/118181224/ed7d0cb9-bb18-4d12-aa23-d123f95d4750">
  
+ Base de datos de vectores en el agente AI
 
  ![imagen](https://github.com/Juc28/TareaLLM-Arep/assets/118181224/3d348efa-dd88-4441-82ad-546cf9e2d5d8)

  
## ARQUITECTURA 
1. La clase *[main.py](https://github.com/Juc28/TareaLLM-Arep/blob/master/main.py)* 
   * Utilizando una biblioteca llamada **langchain** para interactuar con modelos de aprendizaje automático de lenguaje, particularmente un modelo llamado **LLMChain**, que es un modelo de lenguaje generativo preentrenado.
        * Se importarán clases y funciones de la biblioteca langchain como **LLMChain, OpenAI y PromptTemplate.**
        * Para autenticar las solicitudes de servicio de OpenAI, se establece una clave de API de OpenAI en una variable de entorno.
        * Se crea un modelo de llamada que se utilizará para formular preguntas al modelo.Incluye un marcador de posición para la pregunta.
        * Se crea una instancia de OpenAI, un modelo de lenguaje.Se plantea una pregunta particular sobre la teoría de la ciencia de Popper.
        * El LLMChain se ejecuta con la pregunta planteada.La respuesta que generó el modelo se imprime.
          
2. La clase *[memory.py](https://github.com/Juc28/TareaLLM-Arep/blob/master/memory.py)*
   * Utilizando la biblioteca langchain para realizar una variedad de tareas relacionadas con el procesamiento de lenguaje natural (NLP) y la recuperación de información (IR) mediante el uso de modelos de lenguaje generativo y técnicas de recuperación basadas    en vectores semánticos. Aquí hay una explicación detallada:
      * Importación de Biliotecas:
         - bs4 es la abreviatura de BeautifulSoup, una biblioteca de Python que extrae datos de archivos HTML y XML. Se importa aquí para simplificar el procesamiento de documentos HTML.
         - langchain: el módulo principal de la biblioteca langchain se importa.
         - ChatOpenAI, WebBaseLoader, OpenAIEmbeddings, Chroma, StrOutputParser, RunnablePassthrough y RecursiveCharacterTextSplitter son módulos y clases específicos de la comunidad de la cadena de bloques y la cadena de bloques que se utilizan para una               variedad de tareas relacionadas con el procesamiento de lenguaje natural y la recuperación de información.
      * Configuración de las variables de entorno:Para autenticar las solicitudes de servicio de OpenAI, se establece una clave de API de OpenAI en una variable de entorno.
      * Descargar documentos en línea: Un WebBaseLoader utiliza BeautifulSoup (bs4) para parsear el contenido HTML y extraer las secciones pertinentes de la página, y puede cargar documentos desde una URL web específica.
      * Texto dividido:El procesamiento por lotes y la generación de vectores semánticos se facilitan al dividir los documentos cargados en partes más pequeñas con RecursiveCharacterTextSplitter.
      * Construir vectores semánticos:Los vectores semánticos se crean utilizando la técnica de inserción proporcionada por OpenAIEmbeddings y se almacenan en Chroma, un tipo de almacenamiento de vectores semánticos optimizado para la recuperación de              información.
      * Configurar los modelos de chat: ChatOpenAI, un modelo de lenguaje conversacional basado en GPT-3.5, genera respuestas a las preguntas.
      * Configuración de la Cadena de Procesamiento (Pipeline):Se configura un pipeline de procesamiento de texto utilizando diferentes componentes como la recuperación de información, el modelo de lenguaje y un analizador de salida.
      * Invocación del Pipeline:Se utiliza el pipeline configurado para formular una pregunta específica ("What is Task Decomposition?") y obtener una respuesta utilizando el modelo de lenguaje conversacional.
      * Impresión de la Respuesta:La respuesta generada por el modelo de lenguaje se imprime en la consola.
   
3. La clase *[pinecote.py](https://github.com/Juc28/TareaLLM-Arep/blob/master/pinecote.py)*
   * Utiliza la biblioteca langchain y Pinecone, un servicio de indexación y búsqueda de vectores semánticos, para cargar texto, crear vectores semánticos y realizar búsquedas similares a través de documentos.
      - Configuración de variables de entorno y carga de biblioteca:
         * Las clases y funciones de las bibliotecas langchain, langchain_community y pinecone son necesarias.
         * Las claves API de OpenAI y Pinecone se configuran como variables de entorno.
      - La carga de texto es:
         * Para cargar documentos desde un archivo de texto llamado "Conocimiento.txt", se utiliza el TextLoader de langchain_community.
         * El RecursiveCharacterTextSplitter de langchain divide el texto de los documentos en partes más pequeñas.
      - Producir vectores semánticos:
         * Los vectores semánticos para cada fragmento de texto se crean utilizando OpenAIEmbeddings de langchain.
         * Los vectores semánticos creados se almacenan en un índice de Pinecone con PineconeVectorStore.
      - Búsqueda de documentos relacionados:
         * Se establece una consulta de búsqueda utilizando una pregunta particular.
         * Se realiza una búsqueda de similitud utilizando el índice Pinecone previamente creado.
         * Se imprime el contenido del documento más parecido a la pregunta.

# Autor 
Erika Juliana Castro Romero [Juc28](https://github.com/Juc28)



