import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import os
from urllib.parse import quote as urlquote
import base64
from proSimulator import ProDataSimulator

# Load CSV file from Datasets folder

pds = ProDataSimulator(path = "testRoutes", simulate = True)
UPLOAD_DIRECTORY = "testRoutes"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

app = dash.Dash(__name__)


# initializing layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

#home page layout
home_page = html.Div(children=[
    html.H1(className = 'center',children='StravaProSimulator',
        style={
            'textAlign': 'center',
            'color': '#ef3e18'
        }
    ),
    html.Div(children=[
        dcc.Link(
            html.Button(children=[
                html.Img(
                    src=app.get_asset_url('go.png'),
                    style={'width':'30%'}
                ),
                html.Div("Press Go to start",
                         style={'font-family': 'Courier New', 'font-weight': 'bold', 'font-size':'15px', 'color': '#00284d'})],
                style={'background': 'transparent', 'border':'0'}
            ),
            href='/mainPage')
    ], style= {'textAlign': 'center'}),
    html.Div(className='row', children=[
        html.Div(className= 'column', children=[
            html.Div(children=[html.Div(className= 'listHeader', children="About: "),
                               html.Ul(children=[
                                   html.Li(children= "This "),
                               ])],
                     style={'margin-top':'15%', 'font-family': 'Courier New'}
             ),
            html.Div(children=[html.Div(className= 'listHeader', children="The Goal: "),
                               html.Ul(children=[
                                   html.Li(className='nextItem',
                                           children='The ')
                               ])],
                     style={'margin-top': '8%', 'font-family': 'Courier New'}
            )
        ]),
        html.Div(className= 'column', children=[
            html.Div(children=[html.Div(className='listHeader', children="The Elements: "),
                               html.Ul(children=[
                                   html.Li(children= "The elements of the graph page can effectively be broken into three categories"),
                                   html.Li(className='listIndent0',
                                           children= html.P(children=["The ", html.Span(className='listHeader', children="Bar Graph"),
                                                                      ""])
                                           ),
                                   html.Li(className='listIndent0',
                                           children=html.P(
                                               children=["The ", html.Span(className='listHeader', children="6 Filters"),
                                                         ""])
                                           ),
                                   html.Li(className='listIndent0',
                                           children=html.P(
                                               children=["The ", html.Span(className='listHeader', children="Toggle Checkboxes"),
                                                         ""])
                                           )

                               ])],
                     style={'margin-top': '15%', 'font-family': 'Courier New'}
            )
        ])

    ]),
    html.Img(
        src=app.get_asset_url('streaming.png'),
        style={'display': 'block', 'margin-left': 'auto', 'margin-top': '5%','margin-right': 'auto' , 'width':'75%'}
    )


], style={'background-color': '#D3D3D3', 'position':'relative', 'width':'100%', 'height':'100%'})


#main page layout
page_1_layout = html.Div(children=[
    html.H1(children='StravaProSimulator',
            style={
                'textAlign': 'center',
                'color': '#ef3e18'
            }
            ),
    html.Div('Web dashboard for Data Visualization using Python', style={'textAlign': 'center'}),
    html.Div('Streaming Service records', style={'textAlign': 'center'}),
    html.Br(),
    html.Br(),
    dcc.Upload(
            id="upload-data",
            children=html.Div(
                ["Upload gpx file here."]
            ),
            style={
                "width": "20%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            }),
    html.H2("File List"),
    html.Ul(id="file-list")
])



def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    #print(name)
    #data = pds.parseGpx(UPLOAD_DIRECTORY, files = [name])
    #data.to_csv(UPLOAD_DIRECTORY + name, index = False)
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(UPLOAD_DIRECTORY, name), "wb") as fp:
        fp.write(base64.decodebytes(data))

def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files

def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

@app.callback(
    Output("file-list", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def update_output(uploaded_filename, uploaded_file_content):
    """Save uploaded files and regenerate the file list."""

    print(uploaded_filename)

    if uploaded_filename is not None and uploaded_file_content is not None:
        save_file(uploaded_filename, uploaded_file_content)

    files = uploaded_files()
    files.remove(".DS_Store")
    if len(files) == 0:
        return [html.Li("No files yet!")]
    else:
        return [html.Li(file_download_link(filename)) for filename in files]

#callback to change pages
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    print(pathname)
    if pathname == '/mainPage':

        return page_1_layout
    else:
        return home_page       
if __name__ == '__main__':
    app.run_server()




