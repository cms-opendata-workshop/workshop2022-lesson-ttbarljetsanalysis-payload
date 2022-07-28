
# Vector is a Python 3.6+ library for 2D, 3D, and Lorentz vectors,
# especially arrays of vectors, to solve common physics problems in a NumPy-like way.
# https://vector.readthedocs.io/en/latest/
# https://vector.readthedocs.io/en/latest/api/backends/vector.backends.awkward.html?highlight=register_awkward#module-vector.backends.awkward
import vector; vector.register_awkward()

# https://github.com/CoffeaTeam/coffea/blob/master/coffea/nanoevents/transforms.py
from coffea.nanoevents import transforms

# https://coffeateam.github.io/coffea/modules/coffea.nanoevents.methods.base.html
# https://github.com/CoffeaTeam/coffea/blob/master/coffea/nanoevents/methods/base.py
# https://coffeateam.github.io/coffea/modules/coffea.nanoevents.methods.vector.html
# https://github.com/CoffeaTeam/coffea/blob/master/coffea/nanoevents/methods/vector.py
from coffea.nanoevents.methods import base, vector

#https://coffeateam.github.io/coffea/api/coffea.nanoevents.BaseSchema.html
#https://github.com/CoffeaTeam/coffea/blob/master/coffea/nanoevents/schemas/base.py
#https://github.com/CoffeaTeam/coffea/blob/82df7fa06348398346fa365d9f1408fa962e805a/coffea/nanoevents/schemas/base.py#L24
#copare with zip: https://docs.python.org/3/library/functions.html#zip
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms

#-----------------------------
class AGCSchema(BaseSchema):
#-----------------------------
    #-----------------------------
    def __init__(self, base_form):
    #-----------------------------
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])
    #------------------------------------------
    def _build_collections(self, branch_forms):
    #------------------------------------------
        names = set([k.split('_')[0] for k in branch_forms.keys() if not (k.startswith('number'))])
        # Remove n(names) from consideration. It's safe to just remove names that start with n, as nothing else begins with n in our fields.
        # Also remove GenPart, PV and MET because they deviate from the pattern of having a 'number' field.
        names = [k for k in names if not (k.startswith('n') | k.startswith('met') | k.startswith('GenPart') | k.startswith('PV') | k.startswith('trig') | k.startswith('btag'))]
        output = {}
        for name in names:
            #drop branches that are not needed
            offsets = transforms.counts2offsets_form(branch_forms['number' + name])
            #content = {k[len(name)+1:]: branch_forms[k] for k in branch_forms if (k.startswith(name + "_") & (k[len(name)+1:] != 'e'))}
            content = {k[len(name)+1:]: branch_forms[k] for k in branch_forms if (k.startswith(name + "_") & (k[len(name)+1:] != 'e') & (k[len(name)+1:] != 'corre'))}
            # Add energy separately so its treated correctly by the p4 vector.
            # It expects 'energy' and not 'e'
            # https://coffeateam.github.io/coffea/api/coffea.nanoevents.methods.vector.PtEtaPhiELorentzVector.html
            if ( (name == 'jet') or (name == 'fatjet')):
                content['energy'] = branch_forms[name+'_corre']
            else:
                content['energy'] = branch_forms[name+'_e']
            # Check for LorentzVector
            output[name] = zip_forms(content, name, 'PtEtaPhiELorentzVector', offsets=offsets)

        # Handle GenPart, PV, MET, trig and btag. Note that all the nPV_*'s should be the same. We just use one.
        output['met'] = zip_forms({k[len('met')+1:]: branch_forms[k] for k in branch_forms if k.startswith('met_')}, 'met')
        output['trig'] = zip_forms({k[len('trig')+1:]: branch_forms[k] for k in branch_forms if k.startswith('trig_')}, 'trig')
        output['btag'] = zip_forms({k[len('btag')+1:]: branch_forms[k] for k in branch_forms if k.startswith('btag_')}, 'btag')
        #output['GenPart'] = zip_forms({k[len('GenPart')+1:]: branch_forms[k] for k in branch_forms if k.startswith('GenPart_')}, 'GenPart', offsets=transforms.counts2offsets_form(branch_forms['numGenPart']))
        output['PV'] = zip_forms({k[len('PV')+1:]: branch_forms[k] for k in branch_forms if (k.startswith('PV_') & ('npvs' not in k))}, 'PV', offsets=transforms.counts2offsets_form(branch_forms['nPV_x']))
        return output

    @property
    def behavior(self):
        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        return behavior