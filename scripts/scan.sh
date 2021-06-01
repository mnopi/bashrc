sudo journalctl --flush --rotate
sudo rm -rf $HOME/.pen/log/scan.log
sudo touch $HOME/.pen/log/scan.log
sudo chown root:adm $HOME/.pen/log/scan.log
sudo chmod +w $HOME/.pen/log/scan.log
clear
grc tail -f $HOME/.pen/log/scan.log

