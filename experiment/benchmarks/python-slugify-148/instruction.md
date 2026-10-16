Fix uppercase pre-translations

before
```python
>>> from slugify import PRE_TRANSLATIONS
>>> print(PRE_TRANSLATIONS)
[('Ё', 'E'), ('ё', 'e'), ('я', 'ya'), ('х', 'h'), ('у', 'y'), ('щ', 'sch'), ('ю', 'u'), ('Ä', 'Ae'), ('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('Χ', 'Ch'), ('χ', 'ch'), ('Ξ', 'X'), ('ϒ', 'Y'), ('υ', 'y'), ('ύ', 'y'), ('ϋ', 'y'), ('ΰ', 'y')]
```

after

```python
>>> from slugify import PRE_TRANSLATIONS
>>> print(PRE_TRANSLATIONS)
[('Ю', 'U'), ('Щ', 'Sch'), ('У', 'Y'), ('Х', 'H'), ('Я', 'Ya'), ('Ё', 'E'), ('ё', 'e'), ('я', 'ya'), ('х', 'h'), ('у', 'y'), ('щ', 'sch'), ('ю', 'u'), ('Ü', 'Ue'), ('Ö', 'Oe'), ('Ä', 'Ae'), ('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('Ϋ́', 'Y'), ('Ϋ', 'Y'), ('Ύ', 'Y'), ('Υ', 'Y'), ('Χ', 'Ch'), ('χ', 'ch'), ('Ξ', 'X'), ('ϒ', 'Y'), ('υ', 'y'), ('ύ', 'y'), ('ϋ', 'y'), ('ΰ', 'y')]
```
