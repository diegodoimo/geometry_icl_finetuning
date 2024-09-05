dataset = load_from_disk()

subjects = np.array(dataset["subject"])
mask = []
for sub in np.unique(subjects):
    ind = np.nonzero(sub == subjects)[0]
    nsamples = min(samples_per_subject, len(ind))
    chosen = rng.choice(ind, nsamples, replace=False)
    mask.extend(list(np.sort(chosen)))

mask = np.array(mask)
