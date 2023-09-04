
	
		
   BEGIN

		
   DROP VIEW IF EXISTS DUMMY_NON_EXISTENT_VIEW;  






















  




    
        
            
    
        
    
        
            
                
                
   CREATE OR REPLACE VIEW MATERIAL_CHECKOUT_STARTED_CDE80E3A_1524 AS 
SELECT
    *
FROM
    CHECKOUT_STARTED

WHERE
    
        ((
         timestamp <= '2023-09-04T12:50:41.187721Z'
        )
         OR timestamp IS NULL )
    

;  
                
            
        
    

        
    

			
    

   CREATE OR REPLACE TABLE MATERIAL_CHECKOUT_STARTED_VAR_TABLE_4C0E82B2_1524 AS (
        SELECT 
        left(sha1(random()::text),32) AS input_row_id, Material_checkout_started_cde80e3a_1524.*
        FROM Material_checkout_started_cde80e3a_1524);  



			
    
        
            
    
        
    
        
            
                
   DROP VIEW IF EXISTS MATERIAL_CHECKOUT_STARTED_CDE80E3A_1524;  
            
        
    

        
    
 
	
	END;  
	