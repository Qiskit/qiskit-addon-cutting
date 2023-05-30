{#
   This is a copy of the default class template from the sphinx repository,
   with a few minor changes. First, we do not render any inherited members.
   Since we don't generate any inherited members, we include a show-inheritance
   call to give users an indication there may be inherited class members.
   Additionally, we also prevent duplication of the __init__ method by removing it
   from the automethod call in the default.
-#}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
