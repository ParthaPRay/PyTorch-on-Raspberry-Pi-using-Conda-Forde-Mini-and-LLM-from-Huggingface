
# Running a LLM on Raspberry Pi using conda-forge mini, torch and huggingface libraries such as transformers, accelerate, bitsandbytes

* **LLM:** SmolLM-135M-Instruct
* **Conda Package Manager:** Conda-Forge Mini
* **Libraries:**  pytorch, huggingface_hub, transformers, accelerate, bitsandbytes

# Finding Information about Raspberry Pi

1. **Check the OS of the Raspberry Pi**

   ```bash
   $ cat /etc/os-release
   ```

    Output should be somewhat as below:

```bash
	PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
	NAME="Debian GNU/Linux"
	VERSION_ID="12"
	VERSION="12 (bookworm)"
	VERSION_CODENAME=bookworm
	ID=debian
	HOME_URL="https://www.debian.org/"
	SUPPORT_URL="https://www.debian.org/support"
	BUG_REPORT_URL="https://bugs.debian.org/"
```

	This means the Raspberry Pi OS is **“bookworm”**.

2. **Next, check the architecture of Raspberry Pi**

```bash
		$ uname -a
 ```
   
	Output should be somewhat as below:

```bash
	Linux raspberrypi 6.6.28+rpt-rpi-v8 #1 SMP PREEMPT Debian 1:6.6.28-1+rpt1 (2024-04-22) aarch64 GNU/Linux
```
   
	This means the Raspberry Pi architecture is **“aarch64”**. 

3. **Then, check the kernel bit size of Raspberry Pi**

```bash
		$ getconf LONG_BIT
```
	Output should be somewhat as below:
	
```bash
		64
```
	
	This means the Raspberry Pi kernel bit size is **64 bit**.


# Install Conda-Forge Mini

1. **Install conda-forge mini as per https://github.com/conda-forge/miniforge?tab=readme-ov-file**

```bash
	$ wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
```

	**Accept all**

```bash
	$ bash Miniforge3-$(uname)-$(uname -m).sh
```

	Accept all

	**Optional**: If you want to **prevent** Conda from **automatically activating the (base)** environment every time you open a new terminal, you can run: 

 ```bash
	$ conda config --set auto_activate_base false
```
 
	After this step, the conda-forge mini should be successfully installed on Raspberry Pi.


# Directory Formation for Project

1. **Create a directory under which the project should be developed and then cd into it**

	```bash
		$ mkdir smolm
		$ cd smolm
	```

# Conda Virtual Environment Formation

1. **Create the conda virtual environment**

	```bash
	$ conda create -n smolm python=3.10 pip
	```
	This creates a conda virtual environment ‘smolm’ for python 3.10 and pip is also installed for the same python 3.10.

2. **Then activate the conda environment**

	```bash
	$ conda activate smolm
	```

3. **To deactivate the virtual environment for conda do follow:**

	```bash
 	$ conda deactivate
 	```

# PyTorch Installation via Conda-Forge Channel

1. **Install pytorch as per the pytorch web https://pytorch.org/get-started/locally/**

	```bash
 	$ conda install pytorch torchvision torchaudio cpuonly -c pytorch
	```
	If ‘torchaudio’ is** NOT **available then just write below:

	```bash
 	$ conda install pytorch torchvision cpuonly -c pytorch
  	```

2. **Check pytorch version whether it is successfully installed**

2.1. **Option 1:** 

	```bash
 	$ python -c "import torch; print(torch.__version__)"
  	```
   
	OR
 
2.2. **Option 2:**

	```bash
	$ python3
	$ import torch
	$ torch.__version__
	```
	Huggingface Related Libraries Installation via Conda-Forge Channel.

3. **Install ‘huggingface_hub’ from Conda-Forge**

	Use the conda-forge channel to install the huggingface_hub package:

	```bash
 	$ conda install -c conda-forge huggingface_hub
 	```

	**Optional:** After installation, you can verify that huggingface_hub is installed correctly by running:

	```bash
 	$ python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
 	```

	This command fetches information about the ‘GPT-2’ model from Hugging Face Hub, ensuring that the package is working as expected.

2. **Install ‘transformers’ via Conda-Forge**

	```bash
 	$ conda install -c conda-forge transformers
 	```

	**Optional:** Verify the Installation: To verify that the installation was successful, you can try importing the library in Python:

	```bash
 	$ python -c "import transformers; print(transformers.__version__)"
 	```

3. **Install ‘accelerate’ via Conda-Forge**

	```bash
 	$ conda install -c conda-forge accelerate
 	```
	**Optional:** Verify the Installation: To verify that the installation was successful, you can try importing the library in Python:

	```bash
 	$ python -c "import accelerate; print(accelerate.__version__)"
 	```

4. **Install other necessary files via Conda-Forge or pip**


# Optional Step: Install ‘bitsandbytes’ via pip, If Required 

As bitsandbytes is not available for aarch64 platform via Conda-Forge, so use pip3 to install it

	```bash
	$ pip3 install bitsandbytes
 	```

	**Optional:** Verify the Installation: To verify that the installation was successful, you can try importing the library in Python:

	```bash
 	$ python3 -c "import bitsandbytes as bnb; print(bnb.__version__)"
  	```


# Load the LLM from Huggingface and Running

1.** Talk to LLM ‘SmolLM-135M-Instruct’**

	```python
	from transformers import AutoModelForCausalLM, AutoTokenizer
	checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
	
	device = "cpu" # for GPU usage or "cpu" for CPU usage
	tokenizer = AutoTokenizer.from_pretrained(checkpoint)
	# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
	model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
	
	messages = [{"role": "user", "content": "What is the capital of France."}]
	input_text=tokenizer.apply_chat_template(messages, tokenize=False)
	print(input_text)
	inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
	outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
	print(tokenizer.decode(outputs[0]))
	```

# Where about Local Loaded LLMs

1. **Where is Model Loaded in Disk?**

Normally it should be loaded below:

	```bash
 	/home/pi/.cache/huggingface/hub
  	```

**Optional:** To find where the ‘SmolLM-135M-Instruct’ model is loaded on Raspberry Pi run below command:

	```bash
 	$ find / -name "* SmolLM-135M-Instruct *" 2>/dev/null
  	```

2. **How to Remove the ‘SmolLM-135M-Instruct’ From the Disk?**

Based on the above location do follows:

	* Delete Model Directory:

	```bash 
 		$ rm -rf /home/pi/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M-Instruct
  	```

	* Delete Lock File Directory:

	```bash 
 		$ rm -rf /home/pi/.cache/huggingface/hub/.locks/models--HuggingFaceTB--SmolLM-135M-Instruct
   	```

	* Finally, Verify Deletion:

	```bash 
 		$ ls /home/pi/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM-135M-Instruct
   	```

		It should show below:

			```bash 
   				ls: cannot access '/home/pi/.cache/huggingface/hub/models--HuggingFaceTB-- SmolLM-135M-Instruct': No such file or directory
       			```
	```bash 
 		$ ls /home/pi/.cache/huggingface/hub/.locks/models--HuggingFaceTB--SmolLM-135M-Instruct
   	```bash 

		It should show below:

  			```bash 
				ls: cannot access '/home/pi/.cache/huggingface/hub/.locks/models--HuggingFaceTB-- SmolLM-135M-Instruct ': No such file or directory
    			```


