from setuptools import setup
# setup(
#     name='mymodule',
#     packages=['mymodule'],
#     entry_points={
#         'console_scripts' : [
#             'mycommand = mymodule.script:main',
#         ]
#     },
#     install_requires=[
#         'requests',
#     ]
# )
setup(
    name='MLslippage',
    author='jagrio',
    author_email='jagrio@iti.gr',
    packages=['MLslippage'],
    # entry_points={
    #     'console_scripts' : [
    #         'mycommand = ?',
    #     ]
    # },
    install_requires=[
        'numpy>=1.12.0',
        'scipy>=0.18.1',
        'matplotlib>=2.0.0',
        'scikit-learn>=0.18.1',
        'nitime>=0.6',
    ]
)
