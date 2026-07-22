# Project Overview

create from scratch a very basic language model trianed only on philosophy books. This will be done by testing different architectures (transformers) and evalauting responses. The books are downloaded from the gutenbergpy library and stored as .txt files as well as metadata for each book. A full structured pipeline from data extraction and cleaning to a basic frontend where users can interect with the trained model.

## Plotting style
- `plt.style.use("dark_background")`
- Figure size default: `figsize=(10, 6)`
- Figure background (outside plot): `fig.patch.set_facecolor("#000000")`
- Axes background (inside plot): `ax.set_facecolor("#1A1817")`
- Primary plot color (bars/lines/points): `#18D5FFFF`, `alpha=0.95`
- Grid: `ax.grid(True, color="#444444", linewidth=0.8)`
- Spines: `for spine in ax.spines.values(): spine.set_color("#FF8818")`
- Always set a title and axis labels via `ax.set_title()`, `ax.set_xlabel()`, `ax.set_ylabel()`
- Use `plt.tight_layout()` before `plt.show()`/saving

Example:
```python
plt.style.use("dark_background")

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#000000")
ax.set_facecolor("#1A1817")

sns.histplot(data, bins=50, color="#18D5FFFF", ax=ax, alpha=0.95)

ax.grid(True, color="#444444", linewidth=0.8)
for spine in ax.spines.values():
    spine.set_color("#FF8818")

ax.set_title("Title")
ax.set_xlabel("X label")
ax.set_ylabel("Y label")

plt.tight_layout()
plt.show()
```