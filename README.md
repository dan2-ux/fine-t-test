# This repository is for fine tuning mistral model natively on Jetson AGX Orin

### Requirements:
- Hugging face account and token
- Weights & Biases token
- Jetson AGX Orin or Nvidia chip

### **1.** First step login into your hugging-face account by using token

  # Install hugging face cli if not
  
  pip3 install huggingface_cli

  # Login into hugging face cli
  
  huggingface-cli login

### **2.** Install bitsandbytes.
Cloning the required repository
<pre> git clone https://github.com/dusty-nv/jetson-containers.git </pre>
Install the necesary library
<pre>
  # Move to the correct direction
  cd jetson-containers/packages/llm/bitsandbytes
  chmod +x install.sh
  ./install.sh
</pre>

### **3.**  Initiate the jetson-containers enviroment
<pre>
  jetson-containers run -v /path/on/host:/path/in/container $(autotag bitsandbytes)
</pre>
**/path/on/host:**: is your current directory (consider executing **pwd** to get the current directory).

**/path/incontainer**: is the path or folder your will initiate when activating jetson-containers.

### **4.**  Install the necesary library.
<pre> pip install peft trl wandb transformers datasets --index-url https://pypi.org/simple </pre>

### **5.**  Executing the file for fine tune
## Create the python file
<pre>
  # Create your file
  tourch fine-tune-mistral.py
  # Go inside the newly created file and paste all the code in " fine-tune-mistral.py " in
  nano fine-tune-mistral.py
</pre>
## Executing the python file
<pre>
  python3 fine-tune-mistral.py
</pre>

### **6.** Merge and upload it into hugging face.
<pre>
  python3 merge-mistral-upload.py
</pre>

**Warning**: you may need to login into hugging face 

### **7.** Download installed model from hugging face down.
<pre>
  python3 download_fine_tuned.py
</pre>

**Warning**: Copy the path of installed model for later uses.

### **8.** Convert hugging face model into GGUF file
On the same directory as workspace.
<pre>
  mkdir lcpp
  cd lcpp
  git clone https://github.com/ggerganov/llama.cpp.git
  cd llama.cpp
  python3 -m venv .venv
  source ./venv/bin/activate
  pip install -r requirements.txt
</pre>
Then consider running the following command to convert the model
<pre>
  python convert_hf_to_gguf.py /path/to/your/hugging/face/model/that/you/downloaded/
</pre>
For example: /data/models/huggingface/models--dan2-ux--fine-tuned_mistral11/snapshots/83206cf39f9c80f3aea2e7f4f6ca7a0006bfb8a6
Then don't forget to save the directory of the gguf file

### **9.** Enable it to run it
First, return to workspace directory
<pre>
  git clone https://github.com/ggml-org/llama.cpp.git
  cd llama.cpp
  mkdir -p build
  cd build
  apt update
  apt install libcurl4-openssl-dev
  cmake ..
  cmake --build . --parallel
</pre>
To check if sucessfull use **cd bin** if there is huge number of files in it then congrate.

Convert GGUF file into runable model
<pre>
  ./llama-quantize /path/to/your/gguf/file/that/you/have/saved Q4_K_M
</pre>
For example: /data/models/huggingface/models--dan2-ux--fine-tuned_mistral11/snapshots/83206cf39f9c80f3aea2e7f4f6ca7a0006bfb8a6/83206cf39f9c80f3aea2e7f4f6ca7a0006bfb8a6-7.2B-83206cf39f9c80f3aea2e7f4f6ca7a0006bfb8a6-F16.gguf

For testing purposes run the following command
<pre>
  ./llama-cli /path/to/your/gguf/file/that/you/have/saved -p "Hello"
</pre>

### **10.** Running it on Ollama
Run the Modelfile-mistral scripts to create the Modelfile
<pre>
  chmod +x Modelfile-mistral
  sudo ./Modelfile-mistral
</pre>

Create ollam model. However, you need to install ollama for jetson-container first.
<pre>
  ollama create mymodel -f Modelfile
</pre>

