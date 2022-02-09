from setuptools import setup, find_packages

setup(name='faster-rcnn',

      version='0.1',

      url='https://github.com/hongzio/faster-rcnn-tf2',

      license='MIT',

      author='Hongzio',

      author_email='hongzio@user.noreply.github.com',

      description='Faster RCNN implemented in tf2.0',

      packages=find_packages(exclude=['tests']),

      long_description=open('README.md').read(),

      zip_safe=False,

      setup_requires=[],

      install_requires=['tensorflow-gpu==2.5.3', 'PyYAML>=5.4'],

      include_package_data=True
      )
