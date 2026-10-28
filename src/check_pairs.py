import os

real_folder = r"C:\Users\Pawan\Desktop\Final Project\Real-vid"
fake_folder = r"C:\Users\Pawan\Desktop\Final Project\Fake-vid"

real_files = [
    f for f in os.listdir(real_folder)
    if f.lower().endswith(".mp4")
]

fake_files = [
    f for f in os.listdir(fake_folder)
    if f.lower().endswith(".mp4")
]

# Get real video IDs
real_ids = {
    os.path.splitext(f)[0]
    for f in real_files
}

# Check whether fake videos start with a real video ID
matched = 0

for fake in fake_files:
    fake_id = os.path.splitext(fake)[0]
    real_id = fake_id.split("_")[0]

    if real_id in real_ids:
        matched += 1

print("Total real videos:", len(real_files))
print("Total fake videos:", len(fake_files))
print("Fake videos matching a real video ID:", matched)