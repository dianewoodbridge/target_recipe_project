
def dictionary_to_map():
    method_tool_mapping = { 'pot' : '(boil|boiling)' ,
                       'electric oven' : '(bake|baking)',
                       'ladle' : '(stir)',
                       'roaster' :'(roast)',
                       'fry pan' :'(fry)',
                       'colander':'(rinse|drain|drained|strain)',
                       'sifter' : '(sift)',
                       'beater' : '(beat)',
                       'knife' : '(chop|cut)',
                       'chopping board' : '(chop|cut)',
                       'grater' : '(grate)',
                       'spatula' : '(fold)',
                       'saute pan': '(saute)',
                       'griller': '(grill)',
                       'slicer': '(sliced|slice)',
                       'beater': '(beat)',
                       'whisker': '(whisking)'
    }
    
    return method_tool_mapping