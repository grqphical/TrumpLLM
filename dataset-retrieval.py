"""A script to download all the training data for TrumpLLM"""
import os
import shutil
import subprocess

# make sure the API client is installed
if not shutil.which("truthbrush"):
    print("please install truthbrush: https://github.com/stanfordio/truthbrush")
    exit(1)
else:
    print("truthbrush already installed")

username = input("Please enter a TruthSocial username (necessary to pull data): ")
password = input("Please enter a TruthSocial password: ")

env = os.environ.copy()
env["TRUTHSOCIAL_USERNAME"] = username
env["TRUTHSOCIAL_PASSWORD"] = password

with open("foobar.json", "w") as f:
    subprocess.run(["truthbrush", "statuses", "realDonaldTrump"], env=env, shell=True, stdout=f, text=True)