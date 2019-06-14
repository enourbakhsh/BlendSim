#!/bin/bash

start_time="$(date -u +%s)"

# --------------------------
# - file transfer settings
# --------------------------

logfile=screen.txt
source=erfan@emerald.physics.ucdavis.edu:/nfs/home/erfan/airflow/logs/*.logs
dest=/oasis/projects/nsf/ddp340/erfan/rsync_test/

while [ 1 ]
do
    rsync -avz --partial $source $dest --log-file="$logfile"
    if [ "$?" = "0" ] ; then
        echo "rsync completed successfully!"
        break
    else
        echo "Rsync failure. Backing off and retrying..."
        sleep 180
    fi
done

end_time="$(date -u +%s)"
secs="$(($end_time-$start_time))"

# ------------------
# - email settings
# ------------------

subject="File transfer completed successfully"

body=''
body+="Hi Tony and Erfan,\n\n"
body+=$(printf "Congratulations! Your file transfer from NERSC to SDSC has completed successfully in %02d:%02d:%02d hms." $(($secs/3600)) $(($secs%3600/60)) $(($secs%60)))
body+="\n\nSent by Erfan's automated script on GitHub :)"

echo -e "$body" | mailx -s "$subject" -a "$logfile" "erfan@ucdavis.edu"

# used ref:
# https://unix.stackexchange.com/questions/48298/can-rsync-resume-after-being-interrupted
# https://stackoverflow.com/questions/8260858/how-to-send-email-from-terminal
# echo 'mail content' | mailx -s 'email subject' -a attachment.txt username@stackoverflow.com

# usage:
# bash <(wget --delete-after -qO- https://github.com/cosmicshear/BlendSim/raw/master/nersc2sdsc.sh)


