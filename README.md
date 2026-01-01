DiabTrack — Git-Driven Development Story

A Practical Demonstration of Why Git Is Essential in Real ML Systems

Project Context: Why Git Was Necessary from Day One

DiabTrack is not a single-file script. It is a multi-domain system involving:

Data engineering (raw medical datasets)

Model preprocessing & training

Model evaluation & testing

Frontend application (UI)

Configuration & metadata management

Collaboration and review

Each of these components evolves independently, but must eventually work together.
Git was essential to isolate responsibilities, prevent accidental overwrites, and enable safe experimentation.

Step 1: Repository Initialization — Establishing Version Control

Before writing significant logic, Git was initialized to ensure:

Every meaningful change is traceable

Experimental mistakes can be reverted

Multiple development directions can coexist safely

This step marks the transition from “writing code” to “engineering a system”.

Step 2: Structured Commits — Treating Each Domain Separately

As development progressed, different parts of the system were committed independently:

Dataset structure

Model training logic

UI logic

Configuration files

This ensured:

Clear commit history

Logical separation of concerns

Easier debugging and review

<img width="776" height="499" alt="Screenshot (248)" src="https://github.com/user-attachments/assets/c2acc2db-8de4-4596-97a6-b1f01be1590d" />

Step 3: Preprocessing Branch — Isolating a Critical ML Phase

Preprocessing is one of the most sensitive phases in machine learning:

Feature scaling

Encoding

Handling missing values

Preventing data leakage

To avoid destabilizing the main branch, a dedicated preprocessing branch was created.

This allowed:

Aggressive experimentation

Multiple preprocessing strategies

Safe rollback if results degraded

Step 4: Testing Branch — Discovering a Hidden Issue

Once preprocessing was in place, a testing branch was created to evaluate model performance independently.

git checkout -b test/evaluation-suite

<img width="1158" height="550" alt="Screenshot (249)" src="https://github.com/user-attachments/assets/05bd661d-a215-4e1b-a0d1-a6060796e659" />

BRANCHES : 
master	                                Stable, production-ready code
feature/preprocessing-pipeline	        Feature engineering & leakage fixes
test/evaluation-suite	                  Model testing & metrics validation
experiment/alt-ui	UI                    Frontend experimentation
bugfix/documentation-cleanup	          README & documentation fixes

During testing, an important issue surfaced:

A feature caused label leakage, inflating performance metrics.

This discovery validated why testing must be isolated from feature development.

Step 5: Git Stash — Preserving Work Without Polluting History

At this point:

Preprocessing changes were half-complete

Testing revealed a serious issue

Switching branches directly would risk losing work

Instead of committing unstable code, Git stash was used:

git stash push -m "Testing revealed label leakage via preprocessing feature"

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/00c3bbcf-9651-4e4e-80f0-1dee326472f6" />
<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/c8614eda-eae3-49be-90c4-a46709192229" />

This allowed:

Temporary storage of incomplete work

Immediate focus on fixing the testing issue

Clean commit history without experimental noise

Later, the work was safely restored using:

step 6: Collaboration — Inviting a Tester as a Collaborator

To simulate a real-world scenario, a collaborator was invited specifically for testing purposes.

Purpose:

Independent validation

Fresh perspective on evaluation logic

Real collaboration workflow
<img width="1139" height="513" alt="Screenshot 2026-01-01 134003 (1)" src="https://github.com/user-attachments/assets/05ba00f3-61c0-4e11-9ee9-b7970bd1630b" />

The collaborator worked on a separate branch, ensuring:

No direct interference with main code

Controlled scope of contribution

Clean review process

Step 7: Merge Conflict — A Real Collaboration Challenge

During integration, a merge conflict occurred because:

Both contributors modified the same documentation lines

Git could not infer semantic intent

![WhatsApp Image 2026-01-01 at 1 33 25 PM](https://github.com/user-attachments/assets/cd9a0f2f-c68e-4895-949f-cb602da08bdb)

Instead of aborting, the conflict was:

Opened manually

Discussed logically

Resolved by combining both perspectives

Step 8: Model Artifacts & .gitignore — Enforcing ML Hygiene

During training, .pkl model files were generated.

Problem:

Binary files

Large size

Change frequently

Not suitable for version control

This prevents:

Repository bloat

Accidental re-commits

Merge conflicts on binaries

Step 9: Why Git Was Essential for This Project

Without Git:

Preprocessing fixes would overwrite stable models

Testing discoveries would be lost

Collaboration would cause silent conflicts

Model artifacts would pollute history

UI and backend changes would clash

With Git:

Each domain evolved independently

Issues were detected early

Collaboration was structured

History remained clean and meaningful

Git was not used because it was required,
Git was used because the project demanded it.

Challenges Faced & Lessons Learned
Merge Conflicts

Occurred naturally during collaboration

Reinforced importance of communication and review

Label Leakage

Discovered only due to isolated testing branch

Highlighted need for evaluation separation

 Artifact Management

.pkl files initially committed

Corrected using .gitignore and cache removal

Context Switching

Safely handled using Git stash

Each challenge validated why Git workflows exist.
