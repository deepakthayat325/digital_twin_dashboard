import os,sys; print('argv', sys.argv); print('STREAMLIT env', {k:v for k,v in os.environ.items() if 'STREAMLIT' in k}); sys.exit(0)
