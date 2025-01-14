import setuptools

with open("requirements.txt", "r") as file:
    requirements = file.read().splitlines()

setuptools.setup(
    name="valueconsistency",
    version="0.0.1",
    author="Jared Moore",
    author_email="jared@jaredmoore.org",
    description="ValueConsistency",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    entry_points = {
        'console_scripts': [
            'prompt = valueconsistency.prompt:main',
            'judge = valueconsistency.judge:main',
            'run_experiments= valueconsistency.run_experiments:main',
        ]
    },
)
