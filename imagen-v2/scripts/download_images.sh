#! /bin/bash
# https://superuser.com/questions/260092/rsync-switch-to-only-compare-timestamps
# -u
# > This forces rsync to skip any files which exist on the destination and have a modified time 
# > that is newer than the source file. (If an existing destination file has a modification 
# > time equal to the source file’s, it will be updated if the sizes are different.)
rsync -avu --info=progress2 --verbose -z 'rsync://176.9.41.242:873/danbooru2021/original/*' ./data/danbooru/raw/imgs
