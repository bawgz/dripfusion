git clone https://github.com/bawgz/dripfusion

sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog

cd dripfusion
mkdir sdxl-cache
mkdir refiner-cache
wget https://huggingface.co/bawgz/dripfusion/resolve/main/drip_glasses.safetensors --header="Authorization: Bearer $HF_TOKEN" 
wget -c https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar -O - | tar -C ./sdxl-cache/ -xv
wget -c https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar -O - | tar -C ./refiner-cache/ -xv