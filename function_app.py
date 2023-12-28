import azure.functions as func
import logging
import recognition

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="http_trigger")
def http_trigger(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    input_url = req.params.get('input_url')
    if not input_url:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            input_url = req_body.get('input_url')

    httpResponse = None

    if input_url:
        logging.info('Got request with url= ' + input_url)
        try:
            result = recognition.find(input_url)

            if result is not None:
                httpResponse = func.HttpResponse(f"{result}")
        except Exception as e:
            logging.warning(f"Some error happens with {e.args}")
            httpResponse = func.HttpResponse(f"Something bad happens with {input_url}. message: {e.args}", status_code=500)

    else:
        httpResponse = func.HttpResponse("Invalid request parameter", status_code=400)
    
    if httpResponse is None:
        httpResponse = func.HttpResponse(f"Not detected")

    return httpResponse