wget https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
sudo bash script.deb.sh
sudo apt-get install git-lfs
git lfs install
python -m pip install huggingface_hub
huggingface-cli login
git clone https://huggingface.co/bawgz/drip-glasses

sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
