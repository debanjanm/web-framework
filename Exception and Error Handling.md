# Exception and Error Handling

- Introduction
    - Error handling increases the robustness of your code, which guards against potential failures that would cause your program to exit in an uncontrolled fashion.
- Key Differences Between Error and Exception
    - Checked exceptions are generally those from which a program can recover & it might be a good idea to recover from such exceptions programmatically.  A programmer is expected to check for these exceptions by using the try-catch block or throw it back to the caller.
    - On the other hand we have unchecked exceptions. These are those exceptions that might not happen if everything is in order, but they do occur.
    - Errors are also unchecked exception & the programmer is not required to do anything with these. In fact it is a bad idea to use a `try-catch`
    clause for Errors. Most often, recovery from an Error is not possible & the program should be allowed to terminate.
- What are the benefits of Exception Handling?
    - Exception handling makes your code more robust and helps prevent potential failures that would cause your program to stop in an uncontrolled manner.
    - Imagine if you have written a code which is deployed in production and still, it terminates due to an exception, your client would not appreciate that, so it's better to handle the particular exception beforehand and avoid the chaos.
- Built-in Exceptions
    - In Python, all exceptions must be instances of a class that derives from `[BaseException](https://docs.python.org/3/library/exceptions.html#BaseException)`
    .
- Components Of Exception Handling
    - **Try:** It will run the code block in which you expect an error to occur.
    - **Except:** Here, you will define the type of exception you expect in the try block (built-in or custom).
    - **Else:** If there isn't any exception, then this block of code will be executed (consider this as a remedy or a fallback option if you expect a part of your script to produce an exception).
    - **Finally:** Irrespective of whether there is an exception or not, this block of code will always be executed.

- Weblinks
    - **Exceptional logging of exceptions in Python**
        
        [Exceptional Logging of Exceptions in Python](https://www.loggly.com/blog/exceptional-logging-of-exceptions-in-python/)
        
    - The raise statement
        
        [7. Simple statements - Python 3.10.2 documentation](https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement)
        
    - Python Exception Handling Using try, except and finally statement
        
        [Python Exception Handling Using try, except and finally statement](https://www.programiz.com/python-programming/exception-handling)
        
    - Errors and Exceptions
        
        [8. Errors and Exceptions - Python 3.10.2 documentation](https://docs.python.org/3/tutorial/errors.html)
        
    - [https://www.datacamp.com/community/tutorials/exception-handling-python](https://www.datacamp.com/community/tutorials/exception-handling-python)
    - **Python Exception Handling Using try, except and finally statement**
        
        []()
        
    
    ---
    
    - [Manually raising (throwing) an exception in Python](https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python)
    - [How to get out of a try/except inside a while? [Python]](https://stackoverflow.com/questions/3199065/how-to-get-out-of-a-try-except-inside-a-while-python)