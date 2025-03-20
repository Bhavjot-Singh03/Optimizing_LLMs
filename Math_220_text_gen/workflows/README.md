Each phase of the fine tuning process is divided into separate pipelines. 

1. data_pipeline.py : Processes the raw data into the appropriate format for GPT-2 tuning.
2. model_pipeline.py : Performs fine tuning.
3. custom_model_logic : Prepare the model in an format appropriate for langchain RAG.
