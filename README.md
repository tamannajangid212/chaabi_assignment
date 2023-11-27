# chaabi_assignment
<br />
Steps to run:<br />
Download the files.<br />
Download llama2 model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin<br />
In command line write:<br />
pip install -r requirements.txt<br />
<br />
Part 1:<br />
To run streamlit app:<br />
Change the paths used in the code to your respective file paths.<br />
Streamlit run streamllitapp.py<br />
<br />
Part 2:<br />
Change the paths used in the code to your respective file paths.<br />
To run api file:<br />
uvicorn app: app –reload<br />
<br />
Note: app loading and query result processing might take sometime depending upon the system configurations.
Run code on GPU
