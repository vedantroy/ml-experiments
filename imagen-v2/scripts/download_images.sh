#! /bin/bash
rsync -av --info=progress2 --verbose -z 'rsync://176.9.41.242:873/danbooru2021/original/*' ./data/danbooru/raw/imgs
