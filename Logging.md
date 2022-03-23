# Logging

- Introduction
    - Logging is a very useful tool in a programmer’s toolbox. It can help you develop a better understanding of the flow of a program and discover scenarios that you might not even have thought of while developing.
- What are the benefits of Logging?
    - Logs provide developers with an extra set of eyes that are constantly looking at the flow that an application is going through. They can store information, like which user or IP accessed the application. If an error occurs, then they can provide more insights than a stack trace by telling you what the state of the program was before it arrived at the line of code where the error occurred.
    - By logging useful data from the right places, you can not only debug errors easily but also use the data to analyze the performance of the application to plan for scaling or look at usage patterns to plan for marketing.
- The Logging Module
    - The logging module in Python is a ready-to-use and powerful module that is designed to meet the needs of beginners as well as enterprise teams.
    - It is used by most of the third-party Python libraries, so you can integrate your log messages with the ones from those libraries to produce a homogeneous log for your application.
    - With the logging module imported, you can use something called a “logger” to log messages that you want to see. By default, there are 5 standard levels indicating the severity of events. Each has a corresponding method that can be used to log events at that level of severity.
        - DEBUG
        - INFO
        - WARNING
        - ERROR
        - CRITICAL

- Weblinks
    - Development
        - [When to use the different log levels](https://stackoverflow.com/questions/2031163/when-to-use-the-different-log-levels)
        - **Python Logging – Simplest Guide with Full Code and Examples**
            
            [Python Logging - Simplest Guide with Full Code and Examples | ML+](https://www.machinelearningplus.com/python/python-logging-guide/)
            
    - Testing
        - Save the logs generated during a pytest run as a job artifact on GitLab/GitHub CI
            
            [Save the logs generated during a pytest run as a job artifact on GitLab/GitHub CI](https://pawamoy.github.io/posts/save-pytest-logs-as-artifact-gitlab-ci/)
            
        - backtrace=False, but loguru is still showing tracebacks on exceptions
            
            [backtrace=False, but loguru is still showing tracebacks on exceptions · Issue #103 · Delgan/loguru](https://github.com/Delgan/loguru/issues/103)
            
        - "--show-capture=no" option still capture teardown logs.
            
            ["--show-capture=no" option still capture teardown logs. · Issue #3816 · pytest-dev/pytest](https://github.com/pytest-dev/pytest/issues/3816)
            
    - Production
        - Logging in Django Application || Django tutorial 2020 ||
            
            [Logging in Django Application || Django tutorial 2020 ||](https://www.youtube.com/watch?v=-vVml7cpWzY)
            
        - **Logging in Python – How to Use Logs to Debug Your Django Projects**
            
            [Logging in Python - How to Use Logs to Debug Your Django Projects](https://www.freecodecamp.org/news/logging-in-python-debug-your-django-projects/)
            
        - **Django Loguru**
            
            [django-loguru](https://pypi.org/project/django-loguru/)
            
- Libraries
    - **[loguru-caplog 0.2.0](https://pypi.org/project/loguru-caplog/)**
    - **[pytest-logger 0.5.1](https://pypi.org/project/pytest-logger/)**
    - [loguru](https://github.com/Delgan/loguru)
        - How to Add Loguru Support to Pytest in Python
            
            [How to Add Loguru Support to Pytest in Python](https://www.youtube.com/watch?v=eFdVlyAGeZU)
            
        - **The severity levels**
            
            [loguru.logger - loguru documentation](https://loguru.readthedocs.io/en/stable/api/logger.html)