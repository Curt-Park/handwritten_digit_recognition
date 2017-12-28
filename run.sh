python ./vgg16.py --epochs 200
python ../send_sms.py
git add :/
git commit -m 'complete training of vgg16'
git push origin train/vgg16
