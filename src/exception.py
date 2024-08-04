# Here we write about exception handling 
import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):   # error_detail: sys means error_detail is of sys type .
    _,_,exc_tb=error_detail.exe_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message is [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    
    return error_message

class CustomException(Exception):  # CustomException class is inheriting Exception class. Exception class is a base/super class .
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)  # Here super() k/w is calling Exception class whcih will give CustomException class the error msg .
        self.error_message=error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):   # This method converts error message to string .
        return self.error_message    
    
    
# if __name__=='__main__':
#     try:
#         a=1/0
        
#     except Exception as e :
#         logging.info("Divide by Zero !")
#         raise CustomExecption(e,sys)  # This line is calling the constructor which is +nt on line no 13 .



