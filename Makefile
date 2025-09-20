PYTHON = python.exe
DATASET = Result/Dataset
CHECKPOINTS = Result/Checkpoints
DATAZIP ?= dataset.zip

all: dataset train
	@echo "All done.."

# -----------------------------
# Distribution
.distribution:
	$(PYTHON) Distribution.py $(DATASET)
	touch .distribution

distribution: .distribution
	@echo "distribution calculated. See logs for more details"

# -----------------------------
# Augmentation
.augmentation:
	$(PYTHON) Augmentation.py $(DATASET)
	touch .augmentation

augmentation: .augmentation
	@echo "database augmentation done"

# -----------------------------
# Transformation
.transformation:
	$(PYTHON) Transformation.py $(DATASET)
	touch .transformation

transformation: .transformation
	@echo "database transformation done..."

# -----------------------------
# Train
.train:
	$(PYTHON) train.py $(DATASET)
	touch .train

train: .train
	@echo "Model training done..."

# -----------------------------
# Dataset (depends on aug + transform)
.dataset: augmentation transformation
	touch .dataset

dataset: .dataset
	@echo "dataset done"

# -----------------------------
# Reimport dataset
reimport:
	unzip $(DATAZIP) -d $(DATASET)

re: fclean reimport all

# -----------------------------
# Clean targets
clean:
	rm -rf $(DATASET)/*

fclean: clean
	rm -rf .dataset .distribution .augmentation .transformation .train
	rm -rf $(CHECKPOINTS)/*

.PHONY: all clean fclean distribution augmentation transformation train dataset reimport re

