
	
		
   BEGIN

		
   DROP VIEW IF EXISTS DUMMY_NON_EXISTENT_VIEW;  






















  




    
        
            
    
        
    
        
            
                
                
   CREATE OR REPLACE VIEW MATERIAL_CART_CREATE_CDE80E3A_1524 AS 
SELECT
    *
FROM
    CARTS_CREATE

WHERE
    
        ((
         timestamp <= '2023-09-04T12:50:41.187721Z'
        )
         OR timestamp IS NULL )
    

;  
                
            
        
    

        
    

			
    

   CREATE OR REPLACE TABLE MATERIAL_CART_CREATE_VAR_TABLE_4C0E82B2_1524 AS (
        SELECT 
        left(sha1(random()::text),32) AS input_row_id, Material_cart_create_cde80e3a_1524.*
        FROM Material_cart_create_cde80e3a_1524);  



			
    
        
            
    
        
    
        
            
                
   DROP VIEW IF EXISTS MATERIAL_CART_CREATE_CDE80E3A_1524;  
            
        
    

        
    
 
	
	END;  
	