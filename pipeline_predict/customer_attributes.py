from pydantic import BaseModel 

'''
create customer attribute model used for the fastapi connection
'''

class CustomerAttributes(BaseModel): 
    '''
    gets the customer id 
    # instantiate with ** {dictionary}
    # access with CustomerAttributes_instance.key_value 
    '''
    id_number: int =  5008804



