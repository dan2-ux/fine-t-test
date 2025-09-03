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
<pre> pip install peft trl wandb transformers ollama --index-url https://pypi.org/simple </pre>

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

### **5.** Merge and upload it into hugging face.
<pre>
  python3 merge-mistral-upload.py
</pre>

**Warning**: you may need to login into hugging face 
